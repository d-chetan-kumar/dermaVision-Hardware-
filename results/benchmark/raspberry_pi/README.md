# Raspberry Pi 4 benchmark (2026-11-10)

Device: Raspberry Pi 4 Model B Rev 1.5, 4x Cortex-A72 (max 1800 MHz), 4 GB RAM, 64-bit Raspberry Pi OS
(kernel 6.18.50+rpt-rpi-v8), Python 3.13.5, ai-edge-litert 2.3.0 (XNNPACK, CPU only).
Models: SHA-256 identical to `models/*.tflite` on the host.
Protocol: `benchmark.py --model both --e2e --warmup 50 --runs 200`, batch 1, paired alternating.
Input: ISIC_0073082 as 1024x1024 PNG (the host run used the original JPEG; end-to-end times
here include the heavier PNG decode, so compare *pure inference* across devices).
Thermal: 41-56 C, `vcgencmd get_throttled = 0x0` (no throttling) in every run.

| Model | Threads | Pure inference mean (ms) | FPS | End-to-end mean (ms) | Peak RSS (MiB) |
|---|---|---|---|---|---|
| Dynamic-range TFLite | 4 | 40.81 / 41.43 | 24.5 | 131.7 | 52.0 |
| Full INT8 TFLite | 4 | 13.74 / 13.80 | 72.8 | 106.1 | 43.8 |
| Dynamic-range TFLite | 1 | 65.11 | 15.4 | 156.0 | - |
| Full INT8 TFLite | 1 | 45.86 | 21.8 | 138.1 | - |

4-thread values are two independent runs (`threads_4_run1`, `threads_4_run2`).
Peak RSS: whole Python process after 100 invokes (python+numpy+LiteRT baseline = 31.6 MiB).
Accuracy is not re-measured on the Pi: the same .tflite file gives the same predictions;
both models predicted BCC (100%) for the benchmark image, matching the host.
