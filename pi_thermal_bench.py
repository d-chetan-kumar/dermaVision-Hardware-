"""Raspberry Pi FP32 Keras latency + sustained thermal benchmark for DermaVision MobileNetV2.

    python pi_thermal_bench.py latency --image ISIC_0073082.png
    python pi_thermal_bench.py thermal --model int8 --seconds 300

Latency mirrors the host FP32 protocol: batch 1, 50 warm-up + 200 timed runs, preloaded tensor,
time.perf_counter(). Thermal runs back-to-back inference and samples SoC temperature, ARM clock
and the firmware throttle flags (vcgencmd) every --interval seconds.
"""
import argparse, json, resource, statistics, subprocess, time
from pathlib import Path

import numpy as np
from PIL import Image

TFLITE = {"fp32tfl": "models/best_mobilenetv2_fp32.tflite",  # unquantized TFLite, made by `convert`
          "dynamic": "models/best_mobilenetv2.tflite", "int8": "models/best_mobilenetv2_full_int8.tflite"}
KERAS = "models/best_mobilenetv2.keras"


def vc(*args):
    return subprocess.run(["vcgencmd", *args], capture_output=True, text=True).stdout.strip()


def soc_state():
    temp = float(vc("measure_temp").split("=")[1].rstrip("'C"))
    mhz = int(vc("measure_clock", "arm").split("=")[1]) / 1e6
    throttled = int(vc("get_throttled").split("=")[1], 16)
    return temp, mhz, throttled


def load_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return (np.asarray(img, np.float32) / 255.0)[None]


def make_runner(model, x, threads):
    """Return a zero-arg callable that runs one batch-1 inference."""
    if model == "fp32":
        import tensorflow as tf
        m = tf.keras.models.load_model(KERAS, compile=False)
        xt = tf.constant(x)
        return lambda: m(xt, training=False)
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=TFLITE[model], num_threads=threads)
    it.allocate_tensors()
    d, o = it.get_input_details()[0], it.get_output_details()[0]
    if d["dtype"] == np.int8:
        s, z = d["quantization"]
        xin = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
    else:
        xin = x
    def run():
        it.set_tensor(d["index"], xin); it.invoke(); return it.get_tensor(o["index"])
    return run


def latency(args):
    x = load_image(args.image)
    run = make_runner(args.model, x, args.threads)
    for _ in range(args.warmup):
        run()
    t = []
    for _ in range(args.runs):
        t0 = time.perf_counter(); run(); t.append((time.perf_counter() - t0) * 1000)
    probs = np.asarray(run()).ravel()
    res = {"model": args.model, "threads": args.threads, "warmup": args.warmup, "runs": args.runs,
           "mean_ms": statistics.mean(t), "std_ms": statistics.stdev(t), "median_ms": statistics.median(t),
           "min_ms": min(t), "max_ms": max(t), "fps": 1000 / statistics.mean(t),
           "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           "prediction": ["BCC", "Melanoma", "Nevus"][int(probs.argmax())], "confidence": float(probs.max()),
           "disk_mib": Path(KERAS if args.model == "fp32" else TFLITE[args.model]).stat().st_size / 2**20}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, f"latency_{args.model}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def thermal(args):
    x = load_image(args.image)
    run = make_runner(args.model, x, args.threads)
    while soc_state()[0] > args.start_below:  # start every model from the same thermal state
        time.sleep(5)
    for _ in range(20):
        run()
    rows, lat, start = [], [], time.time()
    next_sample = start
    while time.time() - start < args.seconds:
        t0 = time.perf_counter(); run(); lat.append((time.perf_counter() - t0) * 1000)
        if time.time() >= next_sample:
            temp, mhz, thr = soc_state()
            rows.append({"t_s": round(time.time() - start, 1), "temp_c": temp, "arm_mhz": mhz,
                         "throttled_hex": hex(thr), "window_mean_ms": statistics.mean(lat[-50:])})
            next_sample += args.interval
    n = len(lat); first, last = lat[: n // 10], lat[-(n // 10):]
    summary = {"model": args.model, "seconds": args.seconds, "threads": args.threads, "inferences": n,
               "start_temp_c": rows[0]["temp_c"], "max_temp_c": max(r["temp_c"] for r in rows),
               "end_temp_c": rows[-1]["temp_c"], "min_arm_mhz": min(r["arm_mhz"] for r in rows),
               "any_throttle_flag": any(int(r["throttled_hex"], 16) for r in rows),
               "first10pct_mean_ms": statistics.mean(first), "last10pct_mean_ms": statistics.mean(last),
               "overall_mean_ms": statistics.mean(lat)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"thermal_{args.model}.csv", "w") as f:
        f.write(",".join(rows[0]) + "\n")
        f.writelines(",".join(str(v) for v in r.values()) + "\n" for r in rows)
    (out / f"thermal_{args.model}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def convert(args):
    """Plain FP32 TFLite (no quantization) from the Keras checkpoint."""
    import tensorflow as tf
    m = tf.keras.models.load_model(KERAS, compile=False)
    Path(TFLITE["fp32tfl"]).write_bytes(tf.lite.TFLiteConverter.from_keras_model(m).convert())
    print("wrote", TFLITE["fp32tfl"], Path(TFLITE["fp32tfl"]).stat().st_size / 2**20, "MiB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["latency", "thermal", "convert"])
    p.add_argument("--model", choices=["fp32", "fp32tfl", "dynamic", "int8"], default="int8")
    p.add_argument("--image", default="ISIC_0073082.png")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--seconds", type=int, default=300)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--start-below", type=float, default=50.0)
    p.add_argument("--out", default="results/pi_fp32_thermal")
    a = p.parse_args()
    {"latency": latency, "thermal": thermal, "convert": convert}[a.mode](a)
