# DermaVision Edge TFLite Benchmark Report

- Generated At: `2026-11-10 14:33:11`
- Benchmark Protocol: `50` warm-up + `200` timed runs (batch size `1`, CPU execution)
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

## Model Tensor and Quantization Metadata

| Model | File Size (MiB) | Init Time (ms) | Input Shape / Dtype / Scale / ZP | Output Shape / Dtype / Scale / ZP |
|---|---:|---:|---|---|
| `best_mobilenetv2.tflite` | `2.74` | `7.07` | `[1, 224, 224, 3]` / `float32` / `0.0` / `0` | `[1, 3]` / `float32` / `0.0` / `0` |
| `best_mobilenetv2_full_int8.tflite` | `2.94` | `4.23` | `[1, 224, 224, 3]` / `int8` / `0.003921568859368563` / `-128` | `[1, 3]` / `int8` / `0.00390625` / `-128` |

## Benchmark Performance Results

| Model | Mode | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | `148.0285` | `151.9509` | `22.4717` | `68.9106` | `211.8926` | `6.76` |
| `MobileNetV2 Dynamic TFLite` | `end_to_end` | `87.4546` | `84.3701` | `26.3983` | `45.6215` | `201.2664` | `11.43` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | `7.3452` | `7.0002` | `1.4837` | `4.4319` | `13.5701` | `136.14` |
| `MobileNetV2 Full INT8 TFLite` | `end_to_end` | `32.5302` | `30.7937` | `5.9288` | `24.2707` | `61.0137` | `30.74` |

## Output Semantics and Prediction Verification

| Model | Mode | Output Semantics | Predicted Class (Idx) | Confidence (%) | Probabilities |
|---|---|---|---|---:|---|
| `MobileNetV2 Dynamic TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Dynamic TFLite` | `end_to_end` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Full INT8 TFLite` | `pure_inference` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
| `MobileNetV2 Full INT8 TFLite` | `end_to_end` | probabilities (SOFTMAX verified in TFLite graph; softmax not re-applied) | `BCC (0)` | `100.00%` | `{"BCC": 100.0, "Melanoma": 0.0, "Nevus": 0.0}` |
