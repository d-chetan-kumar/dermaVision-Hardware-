"""Standalone loading and CPU inference module for DermaVision DINOv2 ViT-B/14 Dynamic INT8.

Preserves original Phase 1 architecture, 224x224 RGB preprocessing, ImageNet
normalization, and class mapping (0: BCC, 1: Melanoma, 2: Nevus) while exposing
deterministic construction, checkpoint serialization, offline reload, and
single-image prediction helpers without modifying existing application files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ["BCC", "Melanoma", "Nevus"]
CLASS_TO_IDX = {"BCC": 0, "Melanoma": 1, "Nevus": 2}
IDX_TO_CLASS = {0: "BCC", 1: "Melanoma", 2: "Nevus"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)

DEFAULT_HUB_LOCAL_DIR = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
DEFAULT_BACKBONE_PRETRAIN_PATH = (
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
)
DEFAULT_HEAD_CHECKPOINT_PATH = Path("models/dinov2_100_epoch_best.pth")
DEFAULT_INT8_CHECKPOINT_PATH = Path("models/dinov2/dinov2_vitb14_dynamic_int8.pth")


def get_preprocessing_transform() -> transforms.Compose:
    """Return the exact original DINOv2 224x224 RGB ImageNet normalization transform."""
    return transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CombinedDINOv2(nn.Module):
    """DINOv2 ViT-B/14 backbone paired with Linear(768, 3) classification head."""

    def __init__(self, backbone: nn.Module, feature_dim: int = 768, num_classes: int = 3):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.backbone(img)
        return self.classifier(features)


def create_dinov2_vitb14_backbone(
    pretrained: bool = False,
    hub_local_dir: Union[str, Path] = DEFAULT_HUB_LOCAL_DIR,
    pretrain_weights_path: Union[str, Path] = DEFAULT_BACKBONE_PRETRAIN_PATH,
) -> nn.Module:
    """Construct dinov2_vitb14 backbone from local Torch Hub repo (offline-safe)."""
    hub_dir_path = Path(hub_local_dir)
    if hub_dir_path.exists():
        backbone = torch.hub.load(
            str(hub_dir_path),
            "dinov2_vitb14",
            source="local",
            pretrained=False,
            trust_repo=True,
        )
    else:
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=False,
            trust_repo=True,
        )

    if pretrained:
        weights_path = Path(pretrain_weights_path)
        state_dict = torch.load(str(weights_path), map_location="cpu")
        backbone.load_state_dict(state_dict, strict=True)

    backbone.eval()
    return backbone


def load_original_fp32_model(
    head_checkpoint_path: Union[str, Path] = DEFAULT_HEAD_CHECKPOINT_PATH,
    pretrain_weights_path: Union[str, Path] = DEFAULT_BACKBONE_PRETRAIN_PATH,
    hub_local_dir: Union[str, Path] = DEFAULT_HUB_LOCAL_DIR,
) -> Tuple[CombinedDINOv2, Dict[str, Any]]:
    """Reconstruct the original FP32 CombinedDINOv2 model on CPU."""
    backbone = create_dinov2_vitb14_backbone(
        pretrained=True,
        hub_local_dir=hub_local_dir,
        pretrain_weights_path=pretrain_weights_path,
    )
    ckpt = torch.load(str(head_checkpoint_path), map_location="cpu")
    model = CombinedDINOv2(
        backbone=backbone,
        feature_dim=int(ckpt.get("feature_dim", 768)),
        num_classes=int(ckpt.get("num_classes", 3)),
    ).eval()
    model.classifier.load_state_dict(
        {
            "weight": ckpt["model_state_dict"]["classifier.weight"],
            "bias": ckpt["model_state_dict"]["classifier.bias"],
        },
        strict=True,
    )
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


def fold_fixed_224_pos_embed_(backbone: nn.Module, img_size: int = 224) -> torch.Size:
    """EXPERIMENTAL ONLY: Fold bicubic 224x224 interpolated pos_embed into static [1, 257, 768].

    Uses the backbone's original interpolate_pos_encoding() procedure once at 224x224
    so forward() subsequently hits its built-in no-interpolation branch (npatch == N and w == h).
    """
    patch_size = int(getattr(backbone, "patch_size", 14))
    embed_dim = int(getattr(backbone, "embed_dim", 768))
    grid = img_size // patch_size
    dummy_tokens = torch.zeros(1, grid * grid + 1, embed_dim, dtype=torch.float32)
    with torch.no_grad():
        folded = backbone.interpolate_pos_encoding(dummy_tokens, img_size, img_size).detach().clone()
    backbone.pos_embed = nn.Parameter(folded, requires_grad=False)
    return folded.shape


def quantize_model_dynamic_int8(
    fp32_model: CombinedDINOv2,
    engine: str = "onednn",
) -> nn.Module:
    """Apply native PyTorch dynamic INT8 quantization to all nn.Linear modules on CPU."""
    supported = list(getattr(torch.backends.quantized, "supported_engines", []))
    if engine not in supported:
        raise RuntimeError(
            f"Requested quantized engine '{engine}' is not supported. Supported engines: {supported}"
        )
    torch.backends.quantized.engine = engine
    fp32_model.eval()
    q_model = torch.ao.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    q_model.eval()
    return q_model


def build_dynamic_int8_skeleton(
    feature_dim: int = 768,
    num_classes: int = 3,
    fixed_pos_embed_tokens: Optional[int] = None,
    engine: str = "onednn",
    hub_local_dir: Union[str, Path] = DEFAULT_HUB_LOCAL_DIR,
) -> nn.Module:
    """Build an uninitialized dynamic-INT8 CombinedDINOv2 skeleton ready for load_state_dict()."""
    backbone = create_dinov2_vitb14_backbone(
        pretrained=False,
        hub_local_dir=hub_local_dir,
    )
    if fixed_pos_embed_tokens is not None:
        embed_dim = int(getattr(backbone, "embed_dim", 768))
        backbone.pos_embed = nn.Parameter(
            torch.zeros(1, fixed_pos_embed_tokens, embed_dim, dtype=torch.float32),
            requires_grad=False,
        )
    model = CombinedDINOv2(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
    ).eval()
    return quantize_model_dynamic_int8(model, engine=engine)


def save_dynamic_int8_checkpoint(
    q_model: nn.Module,
    output_path: Union[str, Path] = DEFAULT_INT8_CHECKPOINT_PATH,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize quantized state_dict and full configuration metadata to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pos_embed_shape = list(q_model.backbone.pos_embed.shape)
    payload: Dict[str, Any] = {
        "model_state_dict": q_model.state_dict(),
        "model_type": "CombinedDINOv2DynamicINT8",
        "backbone": "dinov2_vitb14",
        "feature_dim": 768,
        "num_classes": 3,
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "input_size": list(INPUT_SIZE),
        "preprocessing": {
            "color_mode": "RGB",
            "resize": list(INPUT_SIZE),
            "interpolation": "BILINEAR",
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
        },
        "pos_embed_shape": pos_embed_shape,
        "pos_embed_behavior": (
            "original_dynamic_interpolate_pos_encoding"
            if pos_embed_shape[1] == 1370
            else f"folded_fixed_{pos_embed_shape[1]}_tokens"
        ),
        "quantization": {
            "method": "torch.ao.quantization.quantize_dynamic",
            "qconfig_spec": ["torch.nn.Linear"],
            "weight_dtype": "torch.qint8",
            "activation_dtype": "torch.quint8 (dynamic per-tensor min/max)",
            "engine": torch.backends.quantized.engine,
            "torch_version": torch.__version__,
        },
        "source_checkpoints": {
            "backbone": "dinov2_vitb14_pretrain.pth",
            "classifier_head": "models/dinov2_100_epoch_best.pth",
        },
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    torch.save(payload, str(out))
    return out


def load_dynamic_int8_model(
    checkpoint_path: Union[str, Path] = DEFAULT_INT8_CHECKPOINT_PATH,
    hub_local_dir: Union[str, Path] = DEFAULT_HUB_LOCAL_DIR,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load the serialized dynamic INT8 DINOv2 model standalone without needing FP32 weights."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    q_info = ckpt.get("quantization", {})
    engine = str(q_info.get("engine", "onednn"))
    pos_shape = ckpt.get("pos_embed_shape", [1, 1370, 768])
    fixed_tokens = int(pos_shape[1]) if int(pos_shape[1]) != 1370 else None

    q_model = build_dynamic_int8_skeleton(
        feature_dim=int(ckpt.get("feature_dim", 768)),
        num_classes=int(ckpt.get("num_classes", 3)),
        fixed_pos_embed_tokens=fixed_tokens,
        engine=engine,
        hub_local_dir=hub_local_dir,
    )
    q_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    q_model.eval()
    return q_model, ckpt


def predict_image(
    q_model: nn.Module,
    image_path: Union[str, Path],
    transform: Optional[transforms.Compose] = None,
) -> Dict[str, Any]:
    """Run single-image inference on a lesion image file and return class + probabilities."""
    tf = transform if transform is not None else get_preprocessing_transform()
    img = Image.open(str(image_path)).convert("RGB")
    tensor = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = q_model(tensor)[0]
        probs = torch.softmax(logits, dim=-1)
    pred_idx = int(torch.argmax(probs).item())
    return {
        "predicted_idx": pred_idx,
        "predicted_class": IDX_TO_CLASS[pred_idx],
        "probabilities": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))},
        "logits": [float(v) for v in logits.tolist()],
    }


if __name__ == "__main__":
    model, metadata = load_dynamic_int8_model()
    print(
        f"Loaded {metadata.get('model_type')} on engine='{torch.backends.quantized.engine}' "
        f"| pos_embed={tuple(model.backbone.pos_embed.shape)}"
    )

