# DermaVision Edge TFLite Benchmark Report

- Generated At: `2026-11-10 16:21:04`
- Benchmark Protocol: `10` warm-up + `50` timed runs (batch size `1`, CPU execution)
- Execution Pairing: `pure_only`
- Threads Requested: `4`
- Input Image Source: `synthetic_512x512_rgb_jpeg`
- Class Mapping: `BCC=0, Melanoma=1, Nevus=2`

## Host Hardware and Runtime Environment

| Field | Measured Value |
|---|---|
| CPU | `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel` |
| Architecture | `AMD64` |
| Logical CPUs | `20` |
| Total RAM (MiB) | `16091.87` |
| Operating System | `Windows 10` |
| Python Version | `3.11.9` |
| TFLite Runtime | `ai-edge-litert (2.3.0)` |
| Delegate Configuration | `dynamic: Built-in delegated op(s) detected (16/83 DELEGATE ops in _get_ops_details; backend not identified by Python API) | int8: Built-in delegated op(s) detected (1/68 DELEGATE ops in _get_ops_details; backend not identified by Python API)` |

> Note: This benchmark ran on a non-Raspberry-Pi host (`Windows 10 / AMD64`). Do not claim on-device Raspberry Pi performance from these host measurements.

## Timing Scope and Parity

- `pure_inference`: Measures `interp.invoke()` only (pre-quantized input tensor copied via `set_tensor` before timer start; `get_tensor` copy and dequantization run after timer stop).
- `end_to_end`: Measures full JPEG open/decode, RGB conversion, `224x224` resize, `[0, 1]` normalization, input quantization, `set_tensor`, `interp.invoke()`, `get_tensor`, output dequantization, and prediction class lookup.
- Paired Execution: When `--e2e` is enabled, both modes share the exact same interpreter instance, thread count, `warmup_runs` count, verified identical input tensor, and alternating `(pure, e2e)` / `(e2e, pure)` iteration schedule to eliminate temporal CPU frequency/load drift.

## Model Tensor and Quantization Metadata

| Model | File Size (MiB) | Init Time (ms) | Input Shape / Dtype / Scale / ZP | Output Shape / Dtype / Scale / ZP |
|---|---:|---:|---|---|
| `best_mobilenetv2.tflite` | `2.74` | `5.60` | `[1, 224, 224, 3]` / `float32` / `0.0` / `0` | `[1, 3]` / `float32` / `0.0` / `0` |
| `best_mobilenetv2_full_int8.tflite` | `2.94` | `3.83` | `[1, 224, 224, 3]` / `int8` / `0.003921568859368563` / `-128` | `[1, 3]` / `int8` / `0.00390625` / `-128` |

## Benchmark Performance Results

| Model | Mode | Mean (ms) | Inner Invoke Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | `92.9301` | `92.9301` | `84.4922` | `53.8046` | `19.0799` | `190.1941` | `10.76` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | `4.1243` | `4.1243` | `4.1642` | `0.8062` | `1.8179` | `5.6941` | `242.47` |

> Threading Note: Multi-subgraph delegation detected on `best_mobilenetv2.tflite` with `threads=4`. On hybrid P/E-core CPUs, fragmented `DELEGATE` boundaries can trigger higher multi-thread synchronization variance than single-subgraph Full INT8 or `--threads 1`.

## Output Semantics and Prediction Verification

| Model | Mode | Output Semantics | Predicted Class (Idx) | Confidence (%) | Probabilities |
|---|---|---|---|---:|---|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `Nevus (2)` | `100.00%` | `{"BCC": 0.0, "Melanoma": 0.0001, "Nevus": 99.9999}` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `Nevus (2)` | `100.00%` | `{"BCC": 0.0, "Melanoma": 0.0, "Nevus": 100.0}` |
