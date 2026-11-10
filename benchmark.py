"""
benchmark.py - DermaVision Edge TFLite Hardware Benchmark
==========================================================
CPU-first benchmarking tool for DermaVision MobileNetV2 TFLite models:
  - Dynamic-Range TFLite: models/best_mobilenetv2.tflite
  - Full Integer INT8 TFLite: models/best_mobilenetv2_full_int8.tflite

Designed for Raspberry Pi and lightweight Linux ARM64/x86_64 edge devices
using ai-edge-litert (preferred) or tflite-runtime, with optional tensorflow
fallback only on development hosts where lightweight runtimes are unavailable.

Usage examples:
    python benchmark.py --model both --e2e
    python benchmark.py --model int8 --warmup 50 --runs 200 --threads 4
    python benchmark.py --model dynamic --e2e --image path/to/lesion.jpg
"""

import argparse
import csv
import datetime
import importlib.metadata as importlib_metadata
import io
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


MODEL_REGISTRY = {
    "dynamic": {
        "key": "dynamic",
        "name": "MobileNetV2 Dynamic TFLite",
        "path": "models/best_mobilenetv2.tflite",
    },
    "int8": {
        "key": "int8",
        "name": "MobileNetV2 Full INT8 TFLite",
        "path": "models/best_mobilenetv2_full_int8.tflite",
    },
}

DEFAULT_CLASS_NAMES_PATH = "models/class_names.json"
EXPECTED_CLASS_MAPPING = {0: "BCC", 1: "Melanoma", 2: "Nevus"}
DEFAULT_OUTPUT_DIR = "results/benchmark"
IMG_SIZE = (224, 224)


def load_tflite_runtime():
    """
    Resolve TFLite Interpreter with priority:
      1. ai-edge-litert
      2. tflite-runtime
      3. tensorflow.lite (optional dev fallback)
    Enforces CPU-only default execution and never imports PyTorch or Streamlit.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    try:
        from ai_edge_litert.interpreter import Interpreter
        try:
            ver = importlib_metadata.version("ai-edge-litert")
        except Exception:
            ver = "unknown"
        return Interpreter, "ai-edge-litert", ver, True
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter
        try:
            ver = importlib_metadata.version("tflite-runtime")
        except Exception:
            ver = "unknown"
        return Interpreter, "tflite-runtime", ver, True
    except ImportError:
        pass

    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("KERAS_HOME", tempfile.gettempdir())
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow-lite-fallback", str(tf.__version__), False
    except ImportError as exc:
        raise RuntimeError(
            "No supported TFLite runtime is installed. Install 'ai-edge-litert' "
            "(preferred for Raspberry Pi / Linux) or 'tflite-runtime'. "
            "On development machines, 'tensorflow' can serve as a fallback."
        ) from exc


def load_and_verify_class_mapping(class_names_path=DEFAULT_CLASS_NAMES_PATH):
    """Load class names JSON and verify canonical mapping BCC=0, Melanoma=1, Nevus=2."""
    path = Path(class_names_path)
    if not path.exists():
        return dict(EXPECTED_CLASS_MAPPING)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping = {int(k): str(v) for k, v in raw.items()}
    if mapping != EXPECTED_CLASS_MAPPING:
        raise ValueError(
            f"Unexpected class mapping in '{class_names_path}': {mapping}. "
            f"Expected {EXPECTED_CLASS_MAPPING}."
        )
    return mapping


def _dtype_name(dtype_obj):
    return np.dtype(dtype_obj).name


def _extract_quant_params(tensor_detail):
    """Extract scalar (scale, zero_point) safely from TFLite tensor details."""
    qp = tensor_detail.get("quantization_parameters") or {}
    scales = qp.get("scales")
    zps = qp.get("zero_points")
    scale = float(scales[0]) if scales is not None and len(scales) > 0 else None
    zp = int(zps[0]) if zps is not None and len(zps) > 0 else None

    if scale is None or zp is None:
        q_tuple = tensor_detail.get("quantization")
        if isinstance(q_tuple, (tuple, list)) and len(q_tuple) == 2:
            t_scale, t_zp = float(q_tuple[0]), int(q_tuple[1])
            if t_scale > 0.0:
                scale = t_scale if scale is None else scale
                zp = t_zp if zp is None else zp

    if scale is None:
        scale = 0.0
    if zp is None:
        zp = 0
    return float(scale), int(zp)


def inspect_delegate_configuration(interpreter):
    """
    Detect explicit delegates or delegated graph ops strictly from runtime APIs.
    Never claims a specific delegate backend without explicit API evidence.
    """
    explicit_delegates = getattr(interpreter, "_delegates", None)
    explicit_count = len(explicit_delegates) if isinstance(explicit_delegates, (list, tuple)) else 0
    explicit_names = [type(d).__name__ for d in explicit_delegates] if explicit_count > 0 else []

    ops_fn = getattr(interpreter, "_get_ops_details", None)
    if callable(ops_fn):
        try:
            ops = ops_fn()
            total_ops = len(ops)
            delegate_ops = sum(1 for op in ops if str(op.get("op_name", "")).upper() == "DELEGATE")
            has_softmax = any(str(op.get("op_name", "")).upper() == "SOFTMAX" for op in ops)
            if explicit_count > 0:
                summary = (
                    f"Explicit delegate(s) configured ({', '.join(explicit_names)}); "
                    f"{delegate_ops}/{total_ops} graph ops delegated"
                )
            elif delegate_ops > 0:
                summary = (
                    f"Built-in delegated op(s) detected ({delegate_ops}/{total_ops} DELEGATE ops "
                    f"in _get_ops_details; backend not identified by Python API)"
                )
            else:
                summary = f"No delegated ops detected ({total_ops} standard CPU ops)"
            return {
                "explicit_delegate_count": explicit_count,
                "explicit_delegates": explicit_names,
                "delegate_ops_count": delegate_ops,
                "total_ops_count": total_ops,
                "graph_has_softmax": has_softmax,
                "detectable": True,
                "summary": summary,
            }
        except Exception:
            pass

    if explicit_count > 0:
        summary = f"Explicit delegate(s) configured ({', '.join(explicit_names)}); op details unavailable"
        detectable = True
    else:
        summary = "Not detectable (runtime Interpreter API does not expose delegate introspection)"
        detectable = False

    return {
        "explicit_delegate_count": explicit_count,
        "explicit_delegates": explicit_names,
        "delegate_ops_count": None,
        "total_ops_count": None,
        "graph_has_softmax": None,
        "detectable": detectable,
        "summary": summary,
    }


def _detect_cpu_name():
    parts = []
    dev_tree_model = Path("/proc/device-tree/model")
    if dev_tree_model.exists():
        try:
            txt = dev_tree_model.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
            if txt:
                parts.append(txt)
        except Exception:
            pass

    proc_cpuinfo = Path("/proc/cpuinfo")
    if proc_cpuinfo.exists():
        try:
            info_lines = proc_cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines()
            for key in ("Model", "model name", "Hardware"):
                for line in info_lines:
                    if line.startswith(key) and ":" in line:
                        val = line.split(":", 1)[1].strip()
                        if val and val not in parts:
                            parts.append(val)
                        break
        except Exception:
            pass

    plat_proc = (platform.processor() or "").strip()
    if plat_proc and plat_proc not in parts:
        parts.append(plat_proc)

    return " | ".join(parts) if parts else (platform.machine() or "Unknown CPU")


def _detect_ram_mib():
    if HAS_PSUTIL:
        return round(psutil.virtual_memory().total / (1024.0 ** 2), 2)
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    kb_str = line.split(":", 1)[1].strip().split()[0]
                    return round(int(kb_str) / 1024.0, 2)
        except Exception:
            pass
    return None


def _detect_available_ram_mib():
    if HAS_PSUTIL:
        return round(psutil.virtual_memory().available / (1024.0 ** 2), 2)
    return None


def _detect_cpu_temp_c():
    temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if temp_path.exists():
        try:
            return round(int(temp_path.read_text(encoding="utf-8").strip()) / 1000.0, 1)
        except Exception:
            pass
    return None


def collect_hardware_info(runtime_name, runtime_version, is_lightweight_rt, threads, models_info):
    total_ram_mib = _detect_ram_mib()
    avail_ram_mib = _detect_available_ram_mib()
    logical_cpus = os.cpu_count()
    if HAS_PSUTIL:
        logical_cpus = psutil.cpu_count(logical=True) or logical_cpus
        phys_cpus = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        freq_mhz = round(float(freq.current), 1) if freq else None
    else:
        phys_cpus = None
        freq_mhz = None

    cpu_name = _detect_cpu_name()
    arch = platform.machine() or "unknown"
    os_name = platform.system()
    is_arm_pi_like = (
        os_name == "Linux"
        and ("raspberry pi" in cpu_name.lower() or arch.lower() in {"aarch64", "armv7l", "armv8l", "arm64"})
    )

    delegate_summaries = []
    any_detectable = False
    for m in models_info:
        d = m["delegate"]
        any_detectable = any_detectable or bool(d["detectable"])
        delegate_summaries.append(f"{m['key']}: {d['summary']}")

    return {
        "cpu_name": cpu_name,
        "architecture": arch,
        "logical_cpu_count": logical_cpus,
        "physical_cpu_count": phys_cpus,
        "cpu_freq_mhz": freq_mhz,
        "cpu_temp_c": _detect_cpu_temp_c(),
        "total_ram_mib": total_ram_mib,
        "available_ram_mib": avail_ram_mib,
        "operating_system": f"{os_name} {platform.release()}",
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "runtime_name": runtime_name,
        "runtime_version": runtime_version,
        "lightweight_edge_runtime": is_lightweight_rt,
        "execution_device": "CPU",
        "thread_count": threads,
        "delegate_configuration": " | ".join(delegate_summaries),
        "delegate_detectable": any_detectable,
        "is_raspberry_pi_or_arm_linux": is_arm_pi_like,
    }


def quantize_input(fp32_batch, input_meta):
    """Quantize [0, 1] FP32 batch to model input dtype using scale and zero-point."""
    dtype_str = input_meta["dtype"]
    scale = input_meta["scale"]
    zp = input_meta["zero_point"]

    if dtype_str == "float32":
        return np.ascontiguousarray(fp32_batch, dtype=np.float32)

    if scale <= 0.0:
        raise ValueError(
            f"Quantized input dtype '{dtype_str}' requires positive scale, got {scale}."
        )

    if dtype_str == "int8":
        q = np.round(fp32_batch / scale + zp)
        return np.clip(q, -128, 127).astype(np.int8)

    if dtype_str == "uint8":
        q = np.round(fp32_batch / scale + zp)
        return np.clip(q, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported input tensor dtype '{dtype_str}'. Supported: float32, int8, uint8.")


def dequantize_output(raw_output, output_meta):
    """Dequantize model output tensor back into FP32 values."""
    dtype_str = output_meta["dtype"]
    scale = output_meta["scale"]
    zp = output_meta["zero_point"]

    arr = np.asarray(raw_output)
    if arr.ndim > 1:
        arr = arr[0]

    if dtype_str == "float32":
        return arr.astype(np.float32)

    if dtype_str in ("int8", "uint8"):
        if scale <= 0.0:
            raise ValueError(
                f"Quantized output dtype '{dtype_str}' requires positive scale, got {scale}."
            )
        return (arr.astype(np.float32) - float(zp)) * float(scale)

    raise ValueError(f"Unsupported output tensor dtype '{dtype_str}'. Supported: float32, int8, uint8.")


def interpret_output_probabilities(dequant_values, output_semantics):
    """
    Convert dequantized output vector into normalized probabilities.
    Never applies softmax twice when model outputs are already probabilities.
    """
    vals = np.asarray(dequant_values, dtype=np.float32).reshape(-1)
    if output_semantics["already_probabilities"]:
        probs = np.clip(vals, 0.0, None)
        total = float(np.sum(probs))
        if total > 0.0 and abs(total - 1.0) > 1e-6:
            probs = probs / total
        return probs

    shifted = vals - float(np.max(vals))
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    return exps / total if total > 0.0 else exps


def detect_output_semantics(dequant_values, graph_has_softmax):
    """Verify whether model output represents probabilities or unnormalized logits."""
    vals = np.asarray(dequant_values, dtype=np.float32).reshape(-1)
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    v_sum = float(np.sum(vals))
    in_prob_range = (-1e-3 <= v_min) and (v_max <= 1.005) and (0.90 <= v_sum <= 1.10)

    if graph_has_softmax is True:
        already_probs = True
        desc = "probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied)"
    elif in_prob_range:
        already_probs = True
        desc = "probabilities (verified in [0, 1] range with sum ~ 1.0; softmax not re-applied)"
    else:
        already_probs = False
        desc = "logits (outside probability simplex; single softmax applied post-dequantization)"

    return {
        "already_probabilities": already_probs,
        "graph_has_softmax": graph_has_softmax,
        "raw_sum": v_sum,
        "description": desc,
    }


def load_model_interpreter(interpreter_cls, model_spec, threads):
    """Initialize TFLite Interpreter cleanly and inspect tensor/quantization metadata."""
    model_path = Path(model_spec["path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Verify path from workspace root."
        )

    size_bytes = model_path.stat().st_size
    size_mib = size_bytes / (1024.0 ** 2)

    t0 = time.perf_counter()
    thread_arg_accepted = True
    try:
        interp = interpreter_cls(model_path=str(model_path), num_threads=int(threads))
    except TypeError:
        interp = interpreter_cls(model_path=str(model_path))
        thread_arg_accepted = False
    interp.allocate_tensors()
    init_time_ms = (time.perf_counter() - t0) * 1000.0

    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = _extract_quant_params(in_d)
    out_scale, out_zp = _extract_quant_params(out_d)

    delegate_info = inspect_delegate_configuration(interp)

    model_info = {
        "key": model_spec["key"],
        "name": model_spec["name"],
        "filename": model_path.name,
        "path": model_path.as_posix(),
        "size_bytes": int(size_bytes),
        "size_mib": round(size_mib, 4),
        "init_time_ms": round(init_time_ms, 3),
        "thread_arg_accepted": thread_arg_accepted,
        "input": {
            "index": int(in_d["index"]),
            "name": str(in_d.get("name", "")),
            "shape": [int(x) for x in in_d["shape"].tolist()],
            "dtype": _dtype_name(in_d["dtype"]),
            "scale": in_scale,
            "zero_point": in_zp,
        },
        "output": {
            "index": int(out_d["index"]),
            "name": str(out_d.get("name", "")),
            "shape": [int(x) for x in out_d["shape"].tolist()],
            "dtype": _dtype_name(out_d["dtype"]),
            "scale": out_scale,
            "zero_point": out_zp,
        },
        "delegate": delegate_info,
    }
    return interp, model_info


def prepare_e2e_source(image_arg):
    """
    Prepare JPEG source for reproducible benchmarking:
      - Uses --image file bytes (or path) if provided.
      - Otherwise generates deterministic synthetic 512x512 RGB JPEG bytes in memory
        so full JPEG decode -> RGB -> resize -> normalize -> quantize works offline.
    """
    if image_arg:
        img_path = Path(image_arg)
        if not img_path.exists():
            raise FileNotFoundError(f"Specified --image file not found: '{img_path}'")
        # Validate decodability immediately before benchmarking
        with Image.open(img_path) as im:
            im.load()
        raw_bytes = img_path.read_bytes()
        return {
            "source_label": img_path.as_posix(),
            "is_synthetic": False,
            "jpeg_bytes": raw_bytes,
            "file_path": img_path,
        }

    rng = np.random.default_rng(42)
    # Deterministic skin-tone-like textured synthetic RGB JPEG (512x512)
    yy, xx = np.mgrid[0:512, 0:512]
    dist = np.sqrt((xx - 256) ** 2 + (yy - 256) ** 2)
    lesion = np.exp(-(dist ** 2) / (2.0 * (85.0 ** 2)))
    noise = rng.normal(0.0, 8.0, size=(512, 512, 3))
    base = np.zeros((512, 512, 3), dtype=np.float32)
    base[..., 0] = 215.0 - 95.0 * lesion
    base[..., 1] = 175.0 - 105.0 * lesion
    base[..., 2] = 155.0 - 100.0 * lesion
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=92)
    return {
        "source_label": "synthetic_512x512_rgb_jpeg",
        "is_synthetic": True,
        "jpeg_bytes": buf.getvalue(),
        "file_path": None,
    }


def preprocess_jpeg_bytes_to_fp32(jpeg_bytes, target_size=IMG_SIZE):
    """Execute full JPEG decode, RGB conversion, 224x224 resize, and [0, 1] normalization."""
    with Image.open(io.BytesIO(jpeg_bytes)) as img:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize(target_size)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def summarize_latencies(latencies_ms):
    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    fps = (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0
    return {
        "mean_ms": round(mean_ms, 4),
        "median_ms": round(median_ms, 4),
        "std_ms": round(std_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "fps": round(fps, 2),
    }


def validate_prediction_record(probs, class_mapping):
    if probs.shape != (len(class_mapping),):
        raise ValueError(f"Expected {len(class_mapping)} outputs, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"Non-finite values in predicted probabilities: {probs}")
    prob_sum = float(np.sum(probs))
    if abs(prob_sum - 1.0) > 1e-3:
        raise ValueError(f"Probabilities must sum to ~1.0, got {prob_sum:.6f} ({probs.tolist()})")

    pred_idx = int(np.argmax(probs))
    pred_class = class_mapping[pred_idx]
    confidence_pct = round(float(probs[pred_idx]) * 100.0, 4)
    probs_pct = {class_mapping[i]: round(float(probs[i]) * 100.0, 4) for i in range(len(class_mapping))}
    return {
        "predicted_index": pred_idx,
        "predicted_class": pred_class,
        "confidence_pct": confidence_pct,
        "probabilities_pct": probs_pct,
        "probability_sum": round(prob_sum, 6),
    }


def _decode_e2e_image_to_fp32(e2e_source, target_hw):
    """Open image path (or JPEG bytes buffer), convert to RGB, resize, and normalize to [0, 1]."""
    file_path = e2e_source["file_path"]
    if file_path is not None:
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize(target_hw)
            arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    else:
        with Image.open(io.BytesIO(e2e_source["jpeg_bytes"])) as img:
            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize(target_hw)
            arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _pure_invoke_step(interp, in_idx, out_idx, prepared_input):
    """
    Time strictly interpreter invocation (`interp.invoke()` only).
    Input tensor assignment (`set_tensor`) precedes the timer start,
    and output reading (`get_tensor`) follows the timer stop.
    """
    interp.set_tensor(in_idx, prepared_input)
    t0 = time.perf_counter()
    interp.invoke()
    invoke_ms = (time.perf_counter() - t0) * 1000.0
    raw_out = np.array(interp.get_tensor(out_idx), copy=True)
    return invoke_ms, raw_out


def _e2e_pipeline_step(interp, model_info, e2e_source, target_hw, class_mapping, known_semantics=None):
    """
    Time full single-image end-to-end pipeline:
      JPEG open/decode -> RGB convert -> resize -> [0, 1] normalization
      -> input quantization -> set_tensor -> invoke -> get_tensor
      -> output dequantization -> probability validation -> class lookup.
    """
    in_meta = model_info["input"]
    out_meta = model_info["output"]
    in_idx = in_meta["index"]
    out_idx = out_meta["index"]
    graph_has_softmax = model_info["delegate"]["graph_has_softmax"]

    t0 = time.perf_counter()
    fp32_batch = _decode_e2e_image_to_fp32(e2e_source, target_hw)
    q_in = quantize_input(fp32_batch, in_meta)
    interp.set_tensor(in_idx, q_in)
    ti0 = time.perf_counter()
    interp.invoke()
    invoke_ms = (time.perf_counter() - ti0) * 1000.0
    raw_out = interp.get_tensor(out_idx)
    dequant = dequantize_output(raw_out, out_meta)
    sem = known_semantics if known_semantics is not None else detect_output_semantics(dequant, graph_has_softmax)
    probs = interpret_output_probabilities(dequant, sem)
    pred_idx = int(np.argmax(probs))
    _ = class_mapping[pred_idx]
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, invoke_ms, sem, probs, q_in


def run_pure_inference_benchmark(interp, model_info, e2e_source, warmup_runs, timed_runs, class_mapping):
    """Benchmark pure batch-1 `interp.invoke()` latency using a pre-quantized input tensor."""
    in_meta = model_info["input"]
    out_meta = model_info["output"]
    target_hw = (in_meta["shape"][2], in_meta["shape"][1])

    fp32_batch = preprocess_jpeg_bytes_to_fp32(e2e_source["jpeg_bytes"], target_size=target_hw)
    prepared_input = quantize_input(fp32_batch, in_meta)

    in_idx = in_meta["index"]
    out_idx = out_meta["index"]
    semantics = None

    for w in range(warmup_runs):
        _, raw_w = _pure_invoke_step(interp, in_idx, out_idx, prepared_input)
        if w == 0:
            dequant_w = dequantize_output(raw_w, out_meta)
            semantics = detect_output_semantics(dequant_w, model_info["delegate"]["graph_has_softmax"])

    latencies = []
    last_raw_out = None
    for _ in range(timed_runs):
        elapsed_ms, last_raw_out = _pure_invoke_step(interp, in_idx, out_idx, prepared_input)
        latencies.append(elapsed_ms)

    dequant = dequantize_output(last_raw_out, out_meta)
    if semantics is None:
        semantics = detect_output_semantics(dequant, model_info["delegate"]["graph_has_softmax"])
    probs = interpret_output_probabilities(dequant, semantics)
    pred = validate_prediction_record(probs, class_mapping)
    stats = summarize_latencies(latencies)
    stats["invoke_only_mean_ms"] = stats["mean_ms"]
    return stats, semantics, pred


def run_end_to_end_benchmark(interp, model_info, e2e_source, warmup_runs, timed_runs, class_mapping):
    """Benchmark standalone end-to-end single-image inference pipeline."""
    in_meta = model_info["input"]
    target_hw = (in_meta["shape"][2], in_meta["shape"][1])
    fixed_semantics = None

    for w in range(warmup_runs):
        _, _, sem_w, _, _ = _e2e_pipeline_step(
            interp, model_info, e2e_source, target_hw, class_mapping, known_semantics=fixed_semantics
        )
        if w == 0:
            fixed_semantics = sem_w

    latencies = []
    invoke_latencies = []
    last_probs = None
    for _ in range(timed_runs):
        total_ms, inv_ms, sem_t, probs, _ = _e2e_pipeline_step(
            interp, model_info, e2e_source, target_hw, class_mapping, known_semantics=fixed_semantics
        )
        if fixed_semantics is None:
            fixed_semantics = sem_t
        latencies.append(total_ms)
        invoke_latencies.append(inv_ms)
        last_probs = probs

    pred = validate_prediction_record(last_probs, class_mapping)
    stats = summarize_latencies(latencies)
    stats["invoke_only_mean_ms"] = round(statistics.mean(invoke_latencies), 4)
    return stats, fixed_semantics, pred


def run_paired_benchmark(interp, model_info, e2e_source, warmup_runs, timed_runs, class_mapping):
    """
    Run paired alternating warm-up and timed measurements for pure_inference and end_to_end
    on the same interpreter instance, thread count, verified identical input tensor, and host state.
    Alternating order on even/odd iterations prevents temporal drift, throttling, or background-load skew.
    """
    in_meta = model_info["input"]
    out_meta = model_info["output"]
    in_idx = in_meta["index"]
    out_idx = out_meta["index"]
    target_hw = (in_meta["shape"][2], in_meta["shape"][1])

    fp32_batch = _decode_e2e_image_to_fp32(e2e_source, target_hw)
    prepared_input = quantize_input(fp32_batch, in_meta)

    fixed_semantics = None
    input_parity_verified = False

    for w in range(warmup_runs):
        if w % 2 == 0:
            _, raw_w = _pure_invoke_step(interp, in_idx, out_idx, prepared_input)
            _, _, sem_w, _, q_e2e = _e2e_pipeline_step(
                interp, model_info, e2e_source, target_hw, class_mapping, known_semantics=fixed_semantics
            )
        else:
            _, _, sem_w, _, q_e2e = _e2e_pipeline_step(
                interp, model_info, e2e_source, target_hw, class_mapping, known_semantics=fixed_semantics
            )
            _, raw_w = _pure_invoke_step(interp, in_idx, out_idx, prepared_input)

        if not input_parity_verified:
            if not np.array_equal(prepared_input, q_e2e):
                raise ValueError("Input tensor mismatch detected between pure_inference and end_to_end modes.")
            input_parity_verified = True
        if fixed_semantics is None:
            dequant_w = dequantize_output(raw_w, out_meta)
            fixed_semantics = detect_output_semantics(dequant_w, model_info["delegate"]["graph_has_softmax"])

    pure_latencies = []
    e2e_latencies = []
    e2e_invoke_latencies = []
    last_raw_out = None
    last_e2e_probs = None

    for i in range(timed_runs):
        if i % 2 == 0:
            p_ms, last_raw_out = _pure_invoke_step(interp, in_idx, out_idx, prepared_input)
            e_ms, e_inv_ms, sem_i, last_e2e_probs, q_e2e = _e2e_pipeline_step(
                interp, model_info, e2e_source, target_hw, class_mapping, known_semantics=fixed_semantics
            )
        else:
            e_ms, e_inv_ms, sem_i, last_e2e_probs, q_e2e = _e2e_pipeline_step(
                interp, model_info, e2e_source, target_hw, class_mapping, known_semantics=fixed_semantics
            )
            p_ms, last_raw_out = _pure_invoke_step(interp, in_idx, out_idx, prepared_input)

        if not input_parity_verified:
            if not np.array_equal(prepared_input, q_e2e):
                raise ValueError("Input tensor mismatch detected between pure_inference and end_to_end modes.")
            input_parity_verified = True
        if fixed_semantics is None:
            fixed_semantics = sem_i

        pure_latencies.append(p_ms)
        e2e_latencies.append(e_ms)
        e2e_invoke_latencies.append(e_inv_ms)

    pure_dequant = dequantize_output(last_raw_out, out_meta)
    pure_probs = interpret_output_probabilities(pure_dequant, fixed_semantics)
    pure_pred = validate_prediction_record(pure_probs, class_mapping)
    pure_stats = summarize_latencies(pure_latencies)
    pure_stats["invoke_only_mean_ms"] = pure_stats["mean_ms"]

    e2e_pred = validate_prediction_record(last_e2e_probs, class_mapping)
    e2e_stats = summarize_latencies(e2e_latencies)
    e2e_stats["invoke_only_mean_ms"] = round(statistics.mean(e2e_invoke_latencies), 4)

    return pure_stats, fixed_semantics, pure_pred, e2e_stats, fixed_semantics, e2e_pred


def build_result_row(
    model_info,
    mode_name,
    stats,
    semantics,
    pred,
    hw_info,
    e2e_source,
    warmup_runs,
    timed_runs,
    threads,
    class_mapping,
    execution_pairing,
):
    in_meta = model_info["input"]
    out_meta = model_info["output"]
    return {
        "model_key": model_info["key"],
        "model_name": model_info["name"],
        "model_filename": model_info["filename"],
        "model_path": model_info["path"],
        "model_size_mib": model_info["size_mib"],
        "benchmark_mode": mode_name,
        "execution_pairing": execution_pairing,
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "threads": int(threads),
        "runtime": f"{hw_info['runtime_name']} ({hw_info['runtime_version']})",
        "delegate_configuration": model_info["delegate"]["summary"],
        "init_time_ms": model_info["init_time_ms"],
        "input_dtype": in_meta["dtype"],
        "input_shape": json.dumps(in_meta["shape"]),
        "input_scale": in_meta["scale"],
        "input_zero_point": in_meta["zero_point"],
        "output_dtype": out_meta["dtype"],
        "output_shape": json.dumps(out_meta["shape"]),
        "output_scale": out_meta["scale"],
        "output_zero_point": out_meta["zero_point"],
        "output_semantics": semantics["description"],
        "mean_ms": stats["mean_ms"],
        "median_ms": stats["median_ms"],
        "std_ms": stats["std_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
        "invoke_only_mean_ms": stats["invoke_only_mean_ms"],
        "fps": stats["fps"],
        "sample_image": e2e_source["source_label"],
        "predicted_class": pred["predicted_class"],
        "predicted_index": pred["predicted_index"],
        "confidence_pct": pred["confidence_pct"],
        "probabilities_pct": json.dumps(pred["probabilities_pct"]),
        "class_mapping": json.dumps(class_mapping),
        "host_os": hw_info["operating_system"],
        "host_cpu": hw_info["cpu_name"],
        "host_arch": hw_info["architecture"],
        "logical_cpus": hw_info["logical_cpu_count"],
        "total_ram_mib": hw_info["total_ram_mib"] if hw_info["total_ram_mib"] is not None else "N/A",
    }


def write_csv_report(rows, csv_path):
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json_report(payload, json_path):
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown_report(payload, md_path):
    hw = payload["hardware"]
    models = payload["models"]
    rows = payload["results"]
    meta = payload["metadata"]

    lines = [
        "# DermaVision Edge TFLite Benchmark Report",
        "",
        f"- Generated At: `{meta['generated_at']}`",
        f"- Benchmark Protocol: `{meta['warmup_runs']}` warm-up + `{meta['timed_runs']}` timed runs (batch size `1`, CPU execution)",
        f"- Execution Pairing: `{meta['execution_pairing']}`",
        f"- Threads Requested: `{hw['thread_count']}`",
        f"- Input Image Source: `{meta['sample_image']}`",
        f"- Class Mapping: `BCC=0, Melanoma=1, Nevus=2`",
        "",
        "## Host Hardware and Runtime Environment",
        "",
        "| Field | Measured Value |",
        "|---|---|",
        f"| CPU | `{hw['cpu_name']}` |",
        f"| Architecture | `{hw['architecture']}` |",
        f"| Logical CPUs | `{hw['logical_cpu_count']}` |",
        f"| Total RAM (MiB) | `{hw['total_ram_mib'] if hw['total_ram_mib'] is not None else 'N/A'}` |",
        f"| Operating System | `{hw['operating_system']}` |",
        f"| Python Version | `{hw['python_version']}` |",
        f"| TFLite Runtime | `{hw['runtime_name']} ({hw['runtime_version']})` |",
        f"| Delegate Configuration | `{hw['delegate_configuration']}` |",
        "",
    ]

    if not hw["is_raspberry_pi_or_arm_linux"]:
        lines.extend([
            "> Note: This benchmark ran on a non-Raspberry-Pi host (`"
            + hw["operating_system"]
            + " / "
            + hw["architecture"]
            + "`). Do not claim on-device Raspberry Pi performance from these host measurements.",
            "",
        ])

    lines.extend([
        "## Timing Scope and Parity",
        "",
        "- `pure_inference`: Measures `interp.invoke()` only (pre-quantized input tensor copied via `set_tensor` before timer start; `get_tensor` copy and dequantization run after timer stop).",
        "- `end_to_end`: Measures full JPEG open/decode, RGB conversion, `224x224` resize, `[0, 1]` normalization, input quantization, `set_tensor`, `interp.invoke()`, `get_tensor`, output dequantization, and prediction class lookup.",
        "- Paired Execution: When `--e2e` is enabled, both modes share the exact same interpreter instance, thread count, `warmup_runs` count, verified identical input tensor, and alternating `(pure, e2e)` / `(e2e, pure)` iteration schedule to eliminate temporal CPU frequency/load drift.",
        "",
        "## Model Tensor and Quantization Metadata",
        "",
        "| Model | File Size (MiB) | Init Time (ms) | Input Shape / Dtype / Scale / ZP | Output Shape / Dtype / Scale / ZP |",
        "|---|---:|---:|---|---|",
    ])
    for m in models:
        inp = m["input"]
        outp = m["output"]
        lines.append(
            f"| `{m['filename']}` | `{m['size_mib']:.2f}` | `{m['init_time_ms']:.2f}` | "
            f"`{inp['shape']}` / `{inp['dtype']}` / `{inp['scale']}` / `{inp['zero_point']}` | "
            f"`{outp['shape']}` / `{outp['dtype']}` / `{outp['scale']}` / `{outp['zero_point']}` |"
        )

    lines.extend([
        "",
        "## Benchmark Performance Results",
        "",
        "| Model | Mode | Mean (ms) | Inner Invoke Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | FPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in rows:
        lines.append(
            f"| `{r['model_name']}` | `{r['benchmark_mode']}` | `{r['mean_ms']:.4f}` | "
            f"`{r['invoke_only_mean_ms']:.4f}` | `{r['median_ms']:.4f}` | `{r['std_ms']:.4f}` | "
            f"`{r['min_ms']:.4f}` | `{r['max_ms']:.4f}` | `{r['fps']:.2f}` |"
        )

    frag_models = [
        m["filename"] for m in models
        if (m["delegate"].get("delegate_ops_count") or 0) > 1 and hw["thread_count"] > 1
    ]
    if frag_models:
        lines.extend([
            "",
            "> Threading Note: Multi-subgraph delegation detected on `"
            + ", ".join(frag_models)
            + "` with `threads="
            + str(hw["thread_count"])
            + "`. On hybrid P/E-core CPUs, fragmented `DELEGATE` boundaries can trigger higher multi-thread synchronization variance than single-subgraph Full INT8 or `--threads 1`.",
        ])

    lines.extend([
        "",
        "## Output Semantics and Prediction Verification",
        "",
        "| Model | Mode | Output Semantics | Predicted Class (Idx) | Confidence (%) | Probabilities |",
        "|---|---|---|---|---:|---|",
    ])
    for r in rows:
        lines.append(
            f"| `{r['model_name']}` | `{r['benchmark_mode']}` | {r['output_semantics']} | "
            f"`{r['predicted_class']} ({r['predicted_index']})` | `{r['confidence_pct']:.2f}%` | `{r['probabilities_pct']}` |"
        )

    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def print_ascii_table(headers, rows, right_align_cols=None):
    right_align_cols = set(right_align_cols or [])
    widths = [len(str(h)) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_row(vals):
        cells = []
        for idx, val in enumerate(vals):
            if idx in right_align_cols:
                cells.append(val.rjust(widths[idx]))
            else:
                cells.append(val.ljust(widths[idx]))
        return " | ".join(cells)

    print(_fmt_row([str(h) for h in headers]))
    print("-+-".join("-" * w for w in widths))
    for row in str_rows:
        print(_fmt_row(row))


def print_terminal_summary(payload, output_paths):
    hw = payload["hardware"]
    models = payload["models"]
    rows = payload["results"]
    meta = payload["metadata"]

    print("")
    print("=" * 84)
    print("DermaVision Edge TFLite Benchmark")
    print("=" * 84)
    print(f"Timestamp      : {meta['generated_at']}")
    print(f"Runtime        : {hw['runtime_name']} ({hw['runtime_version']})")
    print(f"CPU / Arch     : {hw['cpu_name']} [{hw['architecture']}]")
    print(f"OS / Python    : {hw['operating_system']} | Python {hw['python_version']}")
    ram_txt = f"{hw['total_ram_mib']:.1f} MiB" if hw["total_ram_mib"] is not None else "N/A"
    print(f"CPUs / RAM     : {hw['logical_cpu_count']} logical cores | {ram_txt}")
    print(f"Threads / Dev  : {hw['thread_count']} threads | {hw['execution_device']}")
    print(f"Delegate       : {hw['delegate_configuration']}")
    print(f"Protocol       : {meta['warmup_runs']} warmup + {meta['timed_runs']} timed runs ({meta['execution_pairing']}) | Image: {meta['sample_image']}")
    if not hw["is_raspberry_pi_or_arm_linux"]:
        print("Scope Note     : Host is not Raspberry Pi ARM Linux; results reflect local CPU host only.")

    print("")
    print("Model Tensor Metadata")
    print("-" * 84)
    model_rows = []
    for m in models:
        inp = m["input"]
        outp = m["output"]
        model_rows.append([
            m["filename"],
            f"{m['size_mib']:.2f}",
            f"{m['init_time_ms']:.2f}",
            f"{inp['dtype']} {inp['shape']} s={inp['scale']:.6f} z={inp['zero_point']}",
            f"{outp['dtype']} {outp['shape']} s={outp['scale']:.6f} z={outp['zero_point']}",
        ])
    print_ascii_table(
        ["Model", "Size(MiB)", "Init(ms)", "Input Tensor", "Output Tensor"],
        model_rows,
        right_align_cols={1, 2},
    )

    print("")
    print("Latency and Throughput Summary")
    print("-" * 84)
    perf_rows = []
    for r in rows:
        perf_rows.append([
            r["model_name"],
            r["benchmark_mode"],
            f"{r['mean_ms']:.4f}",
            f"{r['invoke_only_mean_ms']:.4f}",
            f"{r['median_ms']:.4f}",
            f"{r['std_ms']:.4f}",
            f"{r['min_ms']:.4f}",
            f"{r['max_ms']:.4f}",
            f"{r['fps']:.2f}",
            f"{r['predicted_class']} ({r['confidence_pct']:.2f}%)",
        ])
    print_ascii_table(
        ["Model", "Mode", "Mean(ms)", "Invoke(ms)", "Med(ms)", "Std(ms)", "Min(ms)", "Max(ms)", "FPS", "Prediction"],
        perf_rows,
        right_align_cols={2, 3, 4, 5, 6, 7, 8},
    )

    print("")
    print("Generated Reports")
    print("-" * 84)
    print(f"CSV      : {output_paths['csv']}")
    print(f"JSON     : {output_paths['json']}")
    print(f"Markdown : {output_paths['markdown']}")
    print("=" * 84)
    print("")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="DermaVision CPU benchmark for Dynamic and Full INT8 MobileNetV2 TFLite models."
    )
    parser.add_argument(
        "--model",
        choices=["dynamic", "int8", "both"],
        default="both",
        help="Model(s) to benchmark: dynamic, int8, or both (default: both).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of untimed warm-up iterations (default: 50).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of timed iterations (default: 200).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads passed to the TFLite interpreter (default: 4).",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Also benchmark full end-to-end JPEG decode + preprocess + quantize + invoke + predict.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a real RGB lesion image for preprocessing and end-to-end benchmarking.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV, JSON, and Markdown reports (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)

    try:
        interpreter_cls, runtime_name, runtime_version, is_lightweight_rt = load_tflite_runtime()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        class_mapping = load_and_verify_class_mapping(DEFAULT_CLASS_NAMES_PATH)
        e2e_source = prepare_e2e_source(args.image)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model_keys = ["dynamic", "int8"] if args.model == "both" else [args.model]
    loaded_models = []
    for key in model_keys:
        spec = MODEL_REGISTRY[key]
        try:
            interp, m_info = load_model_interpreter(interpreter_cls, spec, args.threads)
            loaded_models.append((interp, m_info))
        except Exception as exc:
            print(f"ERROR: Failed to load {spec['name']} ({spec['path']}): {exc}", file=sys.stderr)
            return 1

    hw_info = collect_hardware_info(
        runtime_name=runtime_name,
        runtime_version=runtime_version,
        is_lightweight_rt=is_lightweight_rt,
        threads=args.threads,
        models_info=[m_info for _, m_info in loaded_models],
    )

    execution_pairing = "paired_alternating" if args.e2e else "pure_only"
    result_rows = []
    for interp, m_info in loaded_models:
        if args.e2e:
            pure_stats, pure_sem, pure_pred, e2e_stats, e2e_sem, e2e_pred = run_paired_benchmark(
                interp=interp,
                model_info=m_info,
                e2e_source=e2e_source,
                warmup_runs=args.warmup,
                timed_runs=args.runs,
                class_mapping=class_mapping,
            )
            m_info["output_semantics"] = pure_sem["description"]
            result_rows.append(
                build_result_row(
                    model_info=m_info,
                    mode_name="pure_inference",
                    stats=pure_stats,
                    semantics=pure_sem,
                    pred=pure_pred,
                    hw_info=hw_info,
                    e2e_source=e2e_source,
                    warmup_runs=args.warmup,
                    timed_runs=args.runs,
                    threads=args.threads,
                    class_mapping=class_mapping,
                    execution_pairing=execution_pairing,
                )
            )
            result_rows.append(
                build_result_row(
                    model_info=m_info,
                    mode_name="end_to_end",
                    stats=e2e_stats,
                    semantics=e2e_sem,
                    pred=e2e_pred,
                    hw_info=hw_info,
                    e2e_source=e2e_source,
                    warmup_runs=args.warmup,
                    timed_runs=args.runs,
                    threads=args.threads,
                    class_mapping=class_mapping,
                    execution_pairing=execution_pairing,
                )
            )
        else:
            pure_stats, pure_sem, pure_pred = run_pure_inference_benchmark(
                interp=interp,
                model_info=m_info,
                e2e_source=e2e_source,
                warmup_runs=args.warmup,
                timed_runs=args.runs,
                class_mapping=class_mapping,
            )
            m_info["output_semantics"] = pure_sem["description"]
            result_rows.append(
                build_result_row(
                    model_info=m_info,
                    mode_name="pure_inference",
                    stats=pure_stats,
                    semantics=pure_sem,
                    pred=pure_pred,
                    hw_info=hw_info,
                    e2e_source=e2e_source,
                    warmup_runs=args.warmup,
                    timed_runs=args.runs,
                    threads=args.threads,
                    class_mapping=class_mapping,
                    execution_pairing=execution_pairing,
                )
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_results.csv"
    json_path = out_dir / "benchmark_results.json"
    md_path = out_dir / "benchmark_report.md"

    payload = {
        "metadata": {
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models_requested": args.model,
            "e2e_enabled": bool(args.e2e),
            "execution_pairing": execution_pairing,
            "warmup_runs": int(args.warmup),
            "timed_runs": int(args.runs),
            "threads": int(args.threads),
            "sample_image": e2e_source["source_label"],
            "is_synthetic_image": bool(e2e_source["is_synthetic"]),
            "class_mapping": class_mapping,
        },
        "hardware": hw_info,
        "models": [m_info for _, m_info in loaded_models],
        "results": result_rows,
    }

    write_csv_report(result_rows, csv_path)
    write_json_report(payload, json_path)
    write_markdown_report(payload, md_path)

    print_terminal_summary(
        payload,
        {
            "csv": csv_path.as_posix(),
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
