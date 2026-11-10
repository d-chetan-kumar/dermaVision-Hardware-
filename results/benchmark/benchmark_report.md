# DermaVision Edge TFLite Benchmark Report

- Generated At: `2026-11-10 14:47:57`
- Benchmark Protocol: `50` warm-up + `200` timed runs (batch size `1`, CPU execution)
- Execution Pairing: `paired_alternating`
- Threads Requested: `4`
- Input Image Source: `ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0073082.jpg`
- Class Mapping: `BCC=0, Melanoma=1, Nevus=2`

## Host Hardware and Runtime Environment

| Field | Measured Value |
|---|---|
| CPU | `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel` |
| Architecture | `AMD64` |
| Logical CPUs | `12` |
| Total RAM (MiB) | `16027.69` |
| Operating System | `Windows 11` |
| Python Version | `3.13.5` |
| TFLite Runtime | `tensorflow-lite-fallback (2.21.0)` |
| Delegate Configuration | `dynamic: Built-in delegated op(s) detected (16/83 DELEGATE ops in _get_ops_details; backend not identified by Python API) | int8: Built-in delegated op(s) detected (1/68 DELEGATE ops in _get_ops_details; backend not identified by Python API)` |

> Note: This benchmark ran on a non-Raspberry-Pi host (`Windows 11 / AMD64`). Do not claim on-device Raspberry Pi performance from these host measurements.

## Timing Scope and Parity

- `pure_inference`: Measures `interp.invoke()` only (pre-quantized input tensor copied via `set_tensor` before timer start; `get_tensor` copy and dequantization run after timer stop).
- `end_to_end`: Measures full JPEG open/decode, RGB conversion, `224x224` resize, `[0, 1]` normalization, input quantization, `set_tensor`, `interp.invoke()`, `get_tensor`, output dequantization, and prediction class lookup.
- Paired Execution: When `--e2e` is enabled, both modes share the exact same interpreter instance, thread count, `warmup_runs` count, verified identical input tensor, and alternating `(pure, e2e)` / `(e2e, pure)` iteration schedule to eliminate temporal CPU frequency/load drift.

## Model Tensor and Quantization Metadata

| Model | File Size (MiB) | Init Time (ms) | Input Shape / Dtype / Scale / ZP | Output Shape / Dtype / Scale / ZP |
|---|---:|---:|---|---|
| `best_mobilenetv2.tflite` | `2.74` | `6.15` | `[1, 224, 224, 3]` / `float32` / `0.0` / `0` | `[1, 3]` / `float32` / `0.0` / `0` |
| `best_mobilenetv2_full_int8.tflite` | `2.94` | `4.33` | `[1, 224, 224, 3]` / `int8` / `0.003921568859368563` / `-128` | `[1, 3]` / `int8` / `0.00390625` / `-128` |

## Benchmark Performance Results

| Model | Mode | Mean (ms) | Inner Invoke Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | `109.4814` | `109.4814` | `126.7459` | `52.3582` | `24.5518` | `201.2624` | `9.13` |
| `MobileNetV2 Dynamic TFLite` | `end_to_end` | `133.7434` | `108.0896` | `150.9285` | `53.5949` | `44.0556` | `244.4894` | `7.48` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | `7.2867` | `7.2867` | `6.7877` | `2.1463` | `4.5229` | `20.9802` | `137.24` |
| `MobileNetV2 Full INT8 TFLite` | `end_to_end` | `29.6719` | `7.2324` | `28.7759` | `4.1865` | `23.4003` | `56.6532` | `33.70` |

> Threading Note: Multi-subgraph delegation detected on `best_mobilenetv2.tflite` with `threads=4`. On hybrid P/E-core CPUs, fragmented `DELEGATE` boundaries can trigger higher multi-thread synchronization variance than single-subgraph Full INT8 or `--threads 1`.

## Output Semantics and Prediction Verification

| Model | Mode | Output Semantics | Predicted Class (Idx) | Confidence (%) | Probabilities |
|---|---|---|---|---:|---|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Dynamic TFLite` | `end_to_end` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Full INT8 TFLite` | `end_to_end` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
