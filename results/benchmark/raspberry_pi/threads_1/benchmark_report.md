# DermaVision Edge TFLite Benchmark Report

- Generated At: `2026-11-10 19:14:23`
- Benchmark Protocol: `50` warm-up + `200` timed runs (batch size `1`, CPU execution)
- Execution Pairing: `paired_alternating`
- Threads Requested: `1`
- Input Image Source: `ISIC_0073082.png`
- Class Mapping: `BCC=0, Melanoma=1, Nevus=2`

## Host Hardware and Runtime Environment

| Field | Measured Value |
|---|---|
| CPU | `Raspberry Pi 4 Model B Rev 1.5` |
| Architecture | `aarch64` |
| Logical CPUs | `4` |
| Total RAM (MiB) | `3788.75` |
| Operating System | `Linux 6.18.50+rpt-rpi-v8` |
| Python Version | `3.13.5` |
| TFLite Runtime | `ai-edge-litert (2.3.0)` |
| Delegate Configuration | `dynamic: Built-in delegated op(s) detected (16/83 DELEGATE ops in _get_ops_details; backend not identified by Python API) | int8: Built-in delegated op(s) detected (1/68 DELEGATE ops in _get_ops_details; backend not identified by Python API)` |

## Timing Scope and Parity

- `pure_inference`: Measures `interp.invoke()` only (pre-quantized input tensor copied via `set_tensor` before timer start; `get_tensor` copy and dequantization run after timer stop).
- `end_to_end`: Measures full JPEG open/decode, RGB conversion, `224x224` resize, `[0, 1]` normalization, input quantization, `set_tensor`, `interp.invoke()`, `get_tensor`, output dequantization, and prediction class lookup.
- Paired Execution: When `--e2e` is enabled, both modes share the exact same interpreter instance, thread count, `warmup_runs` count, verified identical input tensor, and alternating `(pure, e2e)` / `(e2e, pure)` iteration schedule to eliminate temporal CPU frequency/load drift.

## Model Tensor and Quantization Metadata

| Model | File Size (MiB) | Init Time (ms) | Input Shape / Dtype / Scale / ZP | Output Shape / Dtype / Scale / ZP |
|---|---:|---:|---|---|
| `best_mobilenetv2.tflite` | `2.74` | `47.27` | `[1, 224, 224, 3]` / `float32` / `0.0` / `0` | `[1, 3]` / `float32` / `0.0` / `0` |
| `best_mobilenetv2_full_int8.tflite` | `2.94` | `18.54` | `[1, 224, 224, 3]` / `int8` / `0.003921568859368563` / `-128` | `[1, 3]` / `int8` / `0.00390625` / `-128` |

## Benchmark Performance Results

| Model | Mode | Mean (ms) | Inner Invoke Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | `65.1145` | `65.1145` | `64.8603` | `0.8039` | `64.4567` | `68.2757` | `15.36` |
| `MobileNetV2 Dynamic TFLite` | `end_to_end` | `155.9628` | `65.1731` | `155.3073` | `2.3438` | `154.4719` | `174.7179` | `6.41` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | `45.8574` | `45.8574` | `45.4797` | `2.1120` | `45.2449` | `66.8646` | `21.81` |
| `MobileNetV2 Full INT8 TFLite` | `end_to_end` | `138.1144` | `45.8022` | `137.3969` | `3.8580` | `136.9224` | `183.3438` | `7.24` |

## Output Semantics and Prediction Verification

| Model | Mode | Output Semantics | Predicted Class (Idx) | Confidence (%) | Probabilities |
|---|---|---|---|---:|---|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Dynamic TFLite` | `end_to_end` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Full INT8 TFLite` | `end_to_end` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
