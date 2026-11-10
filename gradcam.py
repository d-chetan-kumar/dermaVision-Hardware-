import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib as mpl


def preprocess_image_tf(img_path, target_size=(224, 224)):
    """Load and preprocess an image using exact MobileNetV2 training normalization (/ 255.0)."""
    raw = tf.io.read_file(str(img_path))
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, target_size)
    arr = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(arr, axis=0)


def _find_nested_target_layer(model, layer_name):
    """Locate a direct layer or a layer inside a single nested backbone Functional sub-model."""
    direct_names = [layer.name for layer in model.layers]
    if layer_name in direct_names:
        return None, None, model.get_layer(layer_name)

    for idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            inner_names = [sub.name for sub in layer.layers]
            if layer_name in inner_names:
                return idx, layer, layer.get_layer(layer_name)

    raise ValueError(
        f"Layer '{layer_name}' not found in top-level layers {direct_names} or nested sub-models."
    )


def _dense_pre_softmax_logits(layer, x):
    """Compute pre-activation linear logits from a softmax Dense head without mutating the model."""
    return tf.matmul(x, layer.kernel) + (layer.bias if layer.use_bias else 0.0)


def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="out_relu",
    pred_index=None,
    use_pre_softmax=True,
    return_details=False,
):
    """
    Generate a Grad-CAM heatmap supporting nested Functional backbones (e.g., MobileNetV2).

    Uses two-stage nested replay so intermediate feature maps from a wrapped backbone
    sub-model (e.g., mobilenetv2_1.00_224/out_relu) remain connected inside GradientTape.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    backbone_idx, inner_model, target_layer = _find_nested_target_layer(model, last_conv_layer_name)

    if inner_model is None:
        inputs = model.inputs[0] if isinstance(model.inputs, list) and len(model.inputs) == 1 else model.inputs
        final_layer = model.layers[-1]
        use_linear_final = (
            use_pre_softmax
            and isinstance(final_layer, tf.keras.layers.Dense)
            and getattr(final_layer.activation, "__name__", "") == "softmax"
            and target_layer is not final_layer
        )
        if use_linear_final and len(model.layers) >= 2:
            pre_head = tf.keras.models.Model(
                inputs=inputs,
                outputs=[target_layer.output, model.layers[-2].output, model.output],
            )
        else:
            pre_head = tf.keras.models.Model(
                inputs=inputs,
                outputs=[target_layer.output, model.output],
            )
        with tf.GradientTape() as tape:
            if use_linear_final and len(model.layers) >= 2:
                conv_out, penult, preds = pre_head(img_tensor, training=False)
                logits = _dense_pre_softmax_logits(final_layer, penult)
            else:
                conv_out, preds = pre_head(img_tensor, training=False)
                logits = preds
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            target_score = logits[:, pred_index] if use_linear_final else preds[:, pred_index]
        grads = tape.gradient(target_score, conv_out)
    else:
        inner_extractor = tf.keras.Model(
            inputs=inner_model.inputs,
            outputs=[target_layer.output, inner_model.output],
        )
        head_layers = model.layers[backbone_idx + 1 :]
        final_layer = head_layers[-1] if head_layers else None
        is_softmax_dense = (
            use_pre_softmax
            and isinstance(final_layer, tf.keras.layers.Dense)
            and getattr(final_layer.activation, "__name__", "") == "softmax"
        )

        with tf.GradientTape() as tape:
            conv_out, x = inner_extractor(img_tensor, training=False)
            if is_softmax_dense:
                for layer in head_layers[:-1]:
                    x = layer(x, training=False)
                logits = _dense_pre_softmax_logits(final_layer, x)
                preds = tf.nn.softmax(logits, axis=-1)
                score_tensor = logits
            else:
                for layer in head_layers:
                    x = layer(x, training=False)
                preds = x
                score_tensor = x

            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            target_score = score_tensor[:, pred_index]

        grads = tape.gradient(target_score, conv_out)

    if grads is None:
        raise RuntimeError(f"Gradients were None for target layer '{last_conv_layer_name}'.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map = conv_out[0]
    heatmap = tf.squeeze(conv_map @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap_np = np.clip(heatmap.numpy().astype(np.float32), 0.0, 1.0)
    if not return_details:
        return heatmap_np

    return heatmap_np, {
        "probs": preds.numpy()[0].astype(np.float64),
        "logits": (logits.numpy()[0].astype(np.float64) if "logits" in locals() else None),
        "pred_index": int(pred_index),
        "feature_map_shape": tuple(int(d) for d in conv_out.shape),
        "grad_max_abs": float(tf.reduce_max(tf.abs(grads)).numpy()),
        "raw_cam_max": float(max_val.numpy()),
    }


def upsample_heatmap(heatmap, target_width, target_height):
    """Upsample a normalized [0, 1] 2D heatmap cleanly using scalar bicubic interpolation."""
    hm_f = Image.fromarray(np.clip(heatmap, 0.0, 1.0).astype(np.float32), mode="F")
    hm_up = hm_f.resize((int(target_width), int(target_height)), resample=Image.BICUBIC)
    return np.clip(np.asarray(hm_up, dtype=np.float32), 0.0, 1.0)


def colorize_heatmap(heatmap, colormap="jet", target_size=None):
    """Map a normalized [0, 1] heatmap to an RGB PIL.Image using a fixed [0, 1] colormap scale."""
    hm = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    if target_size is not None:
        hm = upsample_heatmap(hm, target_size[0], target_size[1])
    cmap = mpl.colormaps[colormap]
    rgba = cmap(hm)
    rgb = np.uint8(np.clip(rgba[:, :, :3] * 255.0, 0.0, 255.0))
    return Image.fromarray(rgb, mode="RGB")


def overlay_gradcam(img_path, heatmap, alpha=0.4, colormap="jet"):
    """Overlay a fixed-scale Grad-CAM heatmap on the original RGB image."""
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size

    rgb_heatmap_img = colorize_heatmap(heatmap, colormap=colormap, target_size=(img_width, img_height))
    heat_arr = np.asarray(rgb_heatmap_img, dtype=np.float32)
    orig_arr = np.asarray(img, dtype=np.float32)

    superimposed = np.clip(alpha * heat_arr + (1.0 - alpha) * orig_arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(superimposed, mode="RGB")
