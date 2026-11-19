# DermaVision Edge TFLite Benchmark Report

- Generated At: `2026-11-10 19:11:41`
- Benchmark Protocol: `50` warm-up + `200` timed runs (batch size `1`, CPU execution)
- Execution Pairing: `paired_alternating`
- Threads Requested: `4`
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
| `best_mobilenetv2.tflite` | `2.74` | `47.83` | `[1, 224, 224, 3]` / `float32` / `0.0` / `0` | `[1, 3]` / `float32` / `0.0` / `0` |
| `best_mobilenetv2_full_int8.tflite` | `2.94` | `19.10` | `[1, 224, 224, 3]` / `int8` / `0.003921568859368563` / `-128` | `[1, 3]` / `int8` / `0.00390625` / `-128` |

## Benchmark Performance Results

| Model | Mode | Mean (ms) | Inner Invoke Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | `41.4311` | `41.4311` | `40.5487` | `2.9630` | `39.4951` | `66.0385` | `24.14` |
| `MobileNetV2 Dynamic TFLite` | `end_to_end` | `131.8026` | `41.1577` | `131.2836` | `2.0675` | `129.9330` | `146.5901` | `7.59` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | `13.8043` | `13.8043` | `13.7748` | `0.1776` | `13.5549` | `14.9746` | `72.44` |
| `MobileNetV2 Full INT8 TFLite` | `end_to_end` | `106.0596` | `13.8976` | `105.8099` | `0.8831` | `105.3873` | `110.6016` | `9.43` |

> Threading Note: Multi-subgraph delegation detected on `best_mobilenetv2.tflite` with `threads=4`. On hybrid P/E-core CPUs, fragmented `DELEGATE` boundaries can trigger higher multi-thread synchronization variance than single-subgraph Full INT8 or `--threads 1`.

## Output Semantics and Prediction Verification

| Model | Mode | Output Semantics | Predicted Class (Idx) | Confidence (%) | Probabilities |
|---|---|---|---|---:|---|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Dynamic TFLite` | `end_to_end` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Full INT8 TFLite` | `end_to_end` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
