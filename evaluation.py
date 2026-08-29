import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from prediction import load_prediction_model
from utils import load_class_mapping

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

CLASS_NAMES = ["BCC", "Melanoma", "Nevus"]
CLASS_TO_IDX = {"BCC": 0, "Melanoma": 1, "Nevus": 2}
IDX_TO_CLASS = {0: "BCC", 1: "Melanoma", 2: "Nevus"}

# ============================================================
# DATASET UTILS
# ============================================================

def clean_path(path):
    if path is None: return None
    return str(path).strip().strip('"').strip("'")

def build_image_path(image_id, image_directory):
    image_directory = Path(image_directory)
    image_id = str(image_id)
    if image_id.lower().endswith(".jpg"): filename = image_id
    elif image_id.lower().endswith(".jpeg"): filename = image_id
    else: filename = image_id + ".jpg"

    direct_path = image_directory / filename
    if direct_path.exists(): return direct_path

    matches = list(image_directory.rglob(filename))
    if len(matches) > 0: return matches[0]
    return None

def prepare_validation_dataset(ground_truth_csv, image_directory):
    ground_truth_csv = clean_path(ground_truth_csv)
    image_directory = clean_path(image_directory)
    csv_path = Path(ground_truth_csv).expanduser()
    img_dir = Path(image_directory).expanduser()

    if not csv_path.exists(): raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not img_dir.exists(): raise FileNotFoundError(f"Image dir not found: {img_dir}")

    df = pd.read_csv(csv_path)

    mel_df = df[df["MEL"] == 1.0].copy()
    nv_df = df[df["NV"] == 1.0].head(4600).copy()
    bcc_df = df[df["BCC"] == 1.0].copy()

    mel_df["class"] = "Melanoma"
    nv_df["class"] = "Nevus"
    bcc_df["class"] = "BCC"

    selected_df = pd.concat([mel_df, nv_df, bcc_df], ignore_index=True)
    data = selected_df[["image", "class"]].copy()
    data["label"] = data["class"].map(CLASS_TO_IDX)

    train_df, val_df = train_test_split(
        data, test_size=0.20, stratify=data["label"], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    val_df["image_path"] = val_df["image"].apply(lambda x: build_image_path(x, img_dir))
    missing_mask = val_df["image_path"].isna()
    if missing_mask.sum() > 0:
        raise FileNotFoundError(f"{missing_mask.sum()} validation images missing.")
    val_df["image_path"] = val_df["image_path"].apply(str)

    return train_df, val_df


# ============================================================
# MOBILENETV2 (TF) PIPELINE
# ============================================================

def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_validation_tf_dataset(val_df):
    image_paths = val_df["image_path"].values
    labels = val_df["label"].values.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def generate_mobilenet_predictions(model, val_df):
    dataset = create_validation_tf_dataset(val_df)
    y_prob = model.predict(dataset, verbose=1)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_true = val_df["label"].to_numpy(dtype=np.int32)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
    return y_true, y_prob, y_pred


# ============================================================
# DINOV2 (PYTORCH) PIPELINE
# ============================================================

class DINOv2Dataset(torch.utils.data.Dataset):
    def __init__(self, val_df):
        self.val_df = val_df
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.val_df)
        
    def __getitem__(self, idx):
        row = self.val_df.iloc[idx]
        img_path = row['image_path']
        label = row['label']
        image = Image.open(img_path).convert('RGB')
        tensor = self.transform(image)
        return tensor, label

class DINOv2LinearClassifier(nn.Module):
    def __init__(self, feature_dim=768, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

def load_dinov2_model():
    print('Loading DINOv2 backbone...')
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
    backbone.eval()
    
    print('Loading DINOv2 classifier...')
    ckpt = torch.load('models/dinov2_100_epoch_best.pth', map_location='cpu')
    
    classifier = DINOv2LinearClassifier(
        feature_dim=ckpt.get("feature_dim", 768),
        num_classes=ckpt.get("num_classes", 3)
    )
    
    if "model_state_dict" in ckpt:
        classifier.load_state_dict(ckpt["model_state_dict"])
    else:
        classifier.load_state_dict(ckpt)
        
    classifier.eval()
    
    print("DINOv2 classifier loaded successfully")
    print(f"DINOv2 feature dimension: {ckpt.get('feature_dim', 768)}")
    print(f"DINOv2 classes: {ckpt.get('num_classes', 3)}")
    if "best_val_accuracy" in ckpt:
        print(f"DINOv2 checkpoint accuracy: {ckpt['best_val_accuracy']:.4f}")
    if "best_epoch" in ckpt:
        print(f"DINOv2 checkpoint epoch: {ckpt['best_epoch']}")
    
    return backbone, classifier

def generate_dinov2_predictions(backbone, classifier, val_df):
    dataset = DINOv2Dataset(val_df)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    
    y_prob_list = []
    y_true_list = []
    
    print('Running DINOv2 inference...')
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="DINOv2 Batch Progress"):
            images = images.to(device)
            features = backbone(images)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob_list.append(probs)
            y_true_list.append(labels.numpy())
            
    y_prob = np.concatenate(y_prob_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
    return y_true, y_prob, y_pred


# ============================================================
# METRICS CALCULATION
# ============================================================

def calculate_base_metrics(y_true, y_prob, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report_dict = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    auc_values = {}
    fpr_values = {}
    tpr_values = {}
    
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_values[class_name] = float(auc(fpr, tpr))
        fpr_values[class_name] = fpr
        tpr_values[class_name] = tpr
        
    macro_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
    weighted_auc = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
    
    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'auc_bcc': auc_values['BCC'],
        'auc_melanoma': auc_values['Melanoma'],
        'auc_nevus': auc_values['Nevus'],
        'macro_auc': float(macro_auc),
        'weighted_auc': float(weighted_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict,
        'roc': {'fpr': fpr_values, 'tpr': tpr_values}
    }


# ============================================================
# ARTIFACT GENERATION
# ============================================================

def save_artifacts(metrics, prefix, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    report_df.to_csv(out / f'{prefix}_classification_report.csv')
    
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix — {prefix}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out / f'{prefix}_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(9, 7))
    for class_name in CLASS_NAMES:
        fpr = np.array(metrics['roc']['fpr'][class_name])
        tpr = np.array(metrics['roc']['tpr'][class_name])
        auc_value = metrics[f'auc_{class_name.lower()}']
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves — {prefix}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / f'{prefix}_roc_auc.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    classes = CLASS_NAMES
    precisions = [metrics['classification_report'][c]['precision'] for c in classes]
    recalls = [metrics['classification_report'][c]['recall'] for c in classes]
    f1s = [metrics['classification_report'][c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1-score')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(f'Per-Class Performance — {prefix}')
    plt.xticks(x, classes)
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f'{prefix}_per_class_performance.png', dpi=200, bbox_inches='tight')
    plt.close()

def save_model_comparison(metrics_dict, output_dir):
    out = Path(output_dir)
    models = list(metrics_dict.keys())
    
    accs = [metrics_dict[m]['accuracy'] for m in models]
    precs = [metrics_dict[m]['macro_precision'] for m in models]
    recs = [metrics_dict[m]['macro_recall'] for m in models]
    f1s = [metrics_dict[m]['macro_f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, accs, width, label='Accuracy')
    plt.bar(x - 0.5*width, precs, width, label='Macro Precision')
    plt.bar(x + 0.5*width, recs, width, label='Macro Recall')
    plt.bar(x + 1.5*width, f1s, width, label='Macro F1')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, models)
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
def try_save_history(output_dir):
    history_file = Path('metrics/dinov2_training_history.json')
    if not history_file.exists():
        print('\nWARNING: DINOv2 training history file not found.')
        return
        
    with open(history_file, 'r') as f:
        history = json.load(f)
        
    out = Path(output_dir)
    epochs = range(1, len(history['train_accuracy']) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('DINOv2 Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(out / 'dinov2_training_validation_accuracy.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('DINOv2 Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(out / 'dinov2_training_validation_loss.png')
    plt.close()


# ============================================================
# COMPLETE EVALUATION
# ============================================================

def evaluate_model(model=None, ground_truth_csv=None, image_directory=None, output_dir='metrics'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _, val_df = prepare_validation_dataset(ground_truth_csv, image_directory)
    
    assert len(val_df) == 2489, f"CRITICAL ERROR: Expected exactly 2489 validation images, got {len(val_df)}! Halting evaluation."
    
    # 1. MobileNetV2
    print(f"\nEvaluating MobileNetV2: {len(val_df)} validation images")
    mb_model = model if model is not None else load_prediction_model()
    mb_y_true, mb_y_prob, mb_y_pred = generate_mobilenet_predictions(mb_model, val_df)
    mb_metrics = calculate_base_metrics(mb_y_true, mb_y_prob, mb_y_pred)
    
    # 2. DINOv2
    print(f"\nEvaluating DINOv2: {len(val_df)} validation images")
    dino_bb, dino_clf = load_dinov2_model()
    dino_y_true, dino_y_prob, dino_y_pred = generate_dinov2_predictions(dino_bb, dino_clf, val_df)
    dino_metrics = calculate_base_metrics(dino_y_true, dino_y_prob, dino_y_pred)
    
    # Validation Checks
    if len(mb_y_true) != 2489 or len(mb_y_prob) != 2489 or len(dino_y_prob) != 2489:
        raise ValueError("Validation length check failed!")
    if mb_y_prob.shape != (2489, 3) or dino_y_prob.shape != (2489, 3):
        raise ValueError("Probability shape check failed!")
    if not np.array_equal(mb_y_true, dino_y_true):
        raise ValueError("True labels mismatch between MobileNetV2 and DINOv2!")
    if not np.allclose(np.sum(mb_y_prob, axis=1), 1.0, atol=1e-4):
        raise ValueError("MobileNet probabilities do not sum to 1!")
    if not np.allclose(np.sum(dino_y_prob, axis=1), 1.0, atol=1e-4):
        raise ValueError("DINOv2 probabilities do not sum to 1!")
    
    # 3. Ensemble
    print("\nCalculating ensemble metrics...")
    ens_y_prob = 0.5 * mb_y_prob + 0.5 * dino_y_prob
    ens_y_pred = np.argmax(ens_y_prob, axis=1).astype(np.int32)
    ens_metrics = calculate_base_metrics(mb_y_true, ens_y_prob, ens_y_pred)
    
    # Save Artifacts
    print("Generating evaluation graphs...")
    save_artifacts(dino_metrics, 'dinov2', output_dir)
    save_artifacts(ens_metrics, 'ensemble', output_dir)
    save_model_comparison({'MobileNetV2': mb_metrics, 'DINOv2': dino_metrics, 'Ensemble': ens_metrics}, output_dir)
    try_save_history(output_dir)
    
    final_json = {
        'MobileNetV2': {
            'accuracy': mb_metrics['accuracy'],
            'macro_precision': mb_metrics['macro_precision'],
            'macro_recall': mb_metrics['macro_recall'],
            'macro_f1': mb_metrics['macro_f1'],
            'macro_auc': mb_metrics['macro_auc'],
            'weighted_auc': mb_metrics['weighted_auc'],
            'per_class_auc': {
                'BCC': mb_metrics['auc_bcc'],
                'Melanoma': mb_metrics['auc_melanoma'],
                'Nevus': mb_metrics['auc_nevus']
            }
        },
        'DINOv2': {
            'accuracy': dino_metrics['accuracy'],
            'macro_precision': dino_metrics['macro_precision'],
            'macro_recall': dino_metrics['macro_recall'],
            'macro_f1': dino_metrics['macro_f1'],
            'macro_auc': dino_metrics['macro_auc'],
            'weighted_auc': dino_metrics['weighted_auc'],
            'per_class_auc': {
                'BCC': dino_metrics['auc_bcc'],
                'Melanoma': dino_metrics['auc_melanoma'],
                'Nevus': dino_metrics['auc_nevus']
            }
        },
        'Ensemble': {
            'mobilenet_weight': 0.5,
            'dinov2_weight': 0.5,
            'accuracy': ens_metrics['accuracy'],
            'macro_precision': ens_metrics['macro_precision'],
            'macro_recall': ens_metrics['macro_recall'],
            'macro_f1': ens_metrics['macro_f1'],
            'macro_auc': ens_metrics['macro_auc'],
            'weighted_auc': ens_metrics['weighted_auc'],
            'per_class_auc': {
                'BCC': ens_metrics['auc_bcc'],
                'Melanoma': ens_metrics['auc_melanoma'],
                'Nevus': ens_metrics['auc_nevus']
            }
        }
    }
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_json, f, indent=4)
        
    print("\n======================================================================")
    print("FINAL MODEL EVALUATION")
    print("======================================================================")
    print(f"\nMobileNetV2")
    print(f"Accuracy: {mb_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {mb_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {mb_metrics['macro_recall']:.4f}")
    print(f"Macro F1: {mb_metrics['macro_f1']:.4f}")
    print(f"Macro ROC-AUC: {mb_metrics['macro_auc']:.4f}")
    
    print(f"\nDINOv2")
    print(f"Accuracy: {dino_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {dino_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {dino_metrics['macro_recall']:.4f}")
    print(f"Macro F1: {dino_metrics['macro_f1']:.4f}")
    print(f"Macro ROC-AUC: {dino_metrics['macro_auc']:.4f}")
    
    print(f"\nEnsemble")
    print(f"MobileNetV2 weight: 0.5")
    print(f"DINOv2 weight: 0.5")
    print(f"Accuracy: {ens_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {ens_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {ens_metrics['macro_recall']:.4f}")
    print(f"Macro F1: {ens_metrics['macro_f1']:.4f}")
    print(f"Macro ROC-AUC: {ens_metrics['macro_auc']:.4f}")
    
    print("\nGenerated files in metrics/:")
    for file in output_dir.iterdir():
        if file.is_file():
            print(f"- {file.name}")
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to ground truth CSV")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--output_dir", type=str, default="metrics", help="Output metrics directory")
    args = parser.parse_args()
    evaluate_model(model=None, ground_truth_csv=args.csv, image_directory=args.img_dir, output_dir=args.output_dir)