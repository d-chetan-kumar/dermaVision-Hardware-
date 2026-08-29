"""
benchmark.py - DermaVision Hardware Benchmark
==============================================
Measures the real-world performance of the TFLite MobileNetV2 model
on the deployment hardware (Raspberry Pi 4 / any Linux ARM64 device).

Usage:
    python benchmark.py                        # default 50 runs
    python benchmark.py --runs 100             # custom run count
    python benchmark.py --image path/to/img    # use a real image
    python benchmark.py --threads 4            # use 4 CPU threads

Output:
    - Printed report to console
    - benchmark_results.json saved to disk
"""

import argparse
import json
import os
import platform
import sys
import time
import statistics

import numpy as np
from PIL import Image

# -- Optional system metrics --------------------------------------------------
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARNING] psutil not installed -- CPU/RAM metrics will be skipped.")
    print("          Install with: pip install psutil\n")

# -- TFLite Interpreter (ai-edge-litert / tflite-runtime / tensorflow) --------
try:
    from ai_edge_litert.interpreter import Interpreter
    RUNTIME = "ai-edge-litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        RUNTIME = "tflite-runtime"
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        RUNTIME = "tensorflow (fallback)"

# -- Constants ----------------------------------------------------------------
MODEL_PATH  = "models/best_mobilenetv2.tflite"
INPUT_SHAPE = (1, 224, 224, 3)  # MobileNetV2 input


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def section(title):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def row(label, value, unit=""):
    print(f"  {label:<35} {value} {unit}".rstrip())


def make_dummy_image():
    """Creates a random float32 image batch matching MobileNetV2 input."""
    return np.random.rand(*INPUT_SHAPE).astype(np.float32)


def load_real_image(path):
    """Loads, resizes and normalises a real image from disk."""
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def get_ram_mb():
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return 0.0


def get_cpu_temp():
    """Reads CPU temperature on Raspberry Pi."""
    temp_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(temp_path):
        try:
            with open(temp_path) as f:
                return f"{int(f.read().strip()) / 1000:.1f} C"
        except Exception:
            pass
    return "N/A"


def get_system_info():
    info = {
        "platform":      platform.system(),
        "machine":       platform.machine(),
        "processor":     platform.processor() or platform.machine(),
        "python":        sys.version.split()[0],
        "runtime":       RUNTIME,
        "model_path":    MODEL_PATH,
        "model_size_mb": round(os.path.getsize(MODEL_PATH) / (1024 ** 2), 2)
                         if os.path.exists(MODEL_PATH) else "N/A",
    }
    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        info["total_ram_mb"]     = round(vm.total / (1024 ** 2), 1)
        info["available_ram_mb"] = round(vm.available / (1024 ** 2), 1)
        info["cpu_count"]        = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        info["cpu_freq_mhz"]     = round(freq.current, 1) if freq else "N/A"
    info["cpu_temp"] = get_cpu_temp()
    return info


# -----------------------------------------------------------------------------
# Benchmark sections
# -----------------------------------------------------------------------------

def bench_model_load(num_threads):
    """Measures cold model load + tensor allocation time (3 trials)."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        interp = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
        interp.allocate_tensors()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "min_ms":  round(min(times), 2),
        "max_ms":  round(max(times), 2),
        "mean_ms": round(statistics.mean(times), 2),
    }


def bench_inference(interpreter, image_batch, num_runs):
    """
    Runs inference num_runs times.
    First run is warm-up and reported separately.
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    latencies = []
    warmup_ms = 0.0
    for i in range(num_runs + 1):  # +1 for warm-up
        interpreter.set_tensor(input_details[0]["index"], image_batch)
        t0 = time.perf_counter()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _ = interpreter.get_tensor(output_details[0]["index"])

        if i == 0:
            warmup_ms = elapsed_ms
        else:
            latencies.append(elapsed_ms)

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    return {
        "warmup_ms":      round(warmup_ms, 2),
        "min_ms":         round(min(latencies), 2),
        "max_ms":         round(max(latencies), 2),
        "mean_ms":        round(statistics.mean(latencies), 2),
        "median_ms":      round(statistics.median(latencies), 2),
        "stdev_ms":       round(statistics.stdev(latencies), 2),
        "p95_ms":         round(latencies_sorted[int(n * 0.95)], 2),
        "p99_ms":         round(latencies_sorted[int(n * 0.99)], 2),
        "throughput_fps": round(1000 / statistics.mean(latencies), 2),
    }


def bench_preprocessing(image_path, num_runs):
    """Measures image preprocessing pipeline time."""
    times = []
    if image_path and os.path.exists(image_path):
        for _ in range(num_runs):
            t0 = time.perf_counter()
            load_real_image(image_path)
            times.append((time.perf_counter() - t0) * 1000)
        source = os.path.basename(image_path)
    else:
        dummy_pil = Image.fromarray(
            (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        )
        for _ in range(num_runs):
            t0 = time.perf_counter()
            resized = dummy_pil.resize((224, 224))
            arr = np.array(resized).astype(np.float32) / 255.0
            _ = np.expand_dims(arr, axis=0)
            times.append((time.perf_counter() - t0) * 1000)
        source = "synthetic 512x512"

    return {
        "source":    source,
        "min_ms":    round(min(times), 2),
        "max_ms":    round(max(times), 2),
        "mean_ms":   round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
    }


def bench_end_to_end(interpreter, image_batch, num_runs):
    """Full pipeline: set_tensor -> invoke -> get_tensor -> argmax."""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    times = []

    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], image_batch)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]
        _ = int(np.argmax(preds))
        times.append((time.perf_counter() - t0) * 1000)

    n = len(times)
    return {
        "min_ms":         round(min(times), 2),
        "mean_ms":        round(statistics.mean(times), 2),
        "median_ms":      round(statistics.median(times), 2),
        "p95_ms":         round(sorted(times)[int(n * 0.95)], 2),
        "throughput_fps": round(1000 / statistics.mean(times), 2),
    }


def bench_memory(interpreter, image_batch):
    """Snapshot RAM before and after a single inference pass."""
    if not HAS_PSUTIL:
        return {"note": "psutil not available"}

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ram_before = get_ram_mb()
    interpreter.set_tensor(input_details[0]["index"], image_batch)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]["index"])
    ram_after = get_ram_mb()

    return {
        "before_mb": round(ram_before, 1),
        "after_mb":  round(ram_after, 1),
        "delta_mb":  round(ram_after - ram_before, 1),
    }


def bench_cpu_usage(interpreter, image_batch, num_runs):
    """Measures CPU utilisation during a burst of inference calls."""
    if not HAS_PSUTIL:
        return {"note": "psutil not available"}

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    psutil.cpu_percent(interval=None)  # prime the counter
    t0 = time.perf_counter()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]["index"], image_batch)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
    elapsed = time.perf_counter() - t0
    cpu_pct = psutil.cpu_percent(interval=None)

    return {
        "cpu_percent":    round(cpu_pct, 1),
        "total_time_s":   round(elapsed, 3),
        "runs_completed": num_runs,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DermaVision TFLite benchmark for Raspberry Pi")
    p.add_argument("--runs",    type=int, default=50,
                   help="Number of inference runs (default: 50)")
    p.add_argument("--threads", type=int, default=4,
                   help="CPU threads for TFLite (default: 4)")
    p.add_argument("--image",   type=str, default=None,
                   help="Optional path to a real image for preprocessing bench")
    p.add_argument("--output",  type=str, default="benchmark_results.json",
                   help="Path to save JSON results (default: benchmark_results.json)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'")
        print("        Run convert_to_tflite.py first.")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("#  DermaVision TFLite Benchmark")
    print("#" * 60)
    print(f"  Runs     : {args.runs}")
    print(f"  Threads  : {args.threads}")
    print(f"  Runtime  : {RUNTIME}")
    print(f"  Model    : {MODEL_PATH}")

    results = {}

    # System info
    section("System Information")
    sys_info = get_system_info()
    results["system"] = sys_info
    for k, v in sys_info.items():
        row(k.replace("_", " ").title(), v)

    # Model load
    section("Model Load Time  (3 cold trials)")
    load_stats = bench_model_load(args.threads)
    results["model_load"] = load_stats
    row("Min",  load_stats["min_ms"],  "ms")
    row("Mean", load_stats["mean_ms"], "ms")
    row("Max",  load_stats["max_ms"],  "ms")

    # Load interpreter for the remaining tests
    interp = Interpreter(model_path=MODEL_PATH, num_threads=args.threads)
    interp.allocate_tensors()

    # Choose image
    if args.image and os.path.exists(args.image):
        image_batch  = load_real_image(args.image)
        image_source = args.image
    else:
        image_batch  = make_dummy_image()
        image_source = "synthetic random"
    print(f"\n  Image source: {image_source}")

    # Inference latency
    section(f"Inference Latency  ({args.runs} runs, {args.threads} threads)")
    inf_stats = bench_inference(interp, image_batch, args.runs)
    results["inference"] = inf_stats
    row("Warm-up (1st call)",  inf_stats["warmup_ms"],    "ms")
    row("Min",                 inf_stats["min_ms"],        "ms")
    row("Mean",                inf_stats["mean_ms"],       "ms")
    row("Median (P50)",        inf_stats["median_ms"],     "ms")
    row("Std Dev",             inf_stats["stdev_ms"],      "ms")
    row("P95",                 inf_stats["p95_ms"],        "ms")
    row("P99",                 inf_stats["p99_ms"],        "ms")
    row("Max",                 inf_stats["max_ms"],        "ms")
    row("Throughput",          inf_stats["throughput_fps"], "FPS")

    # Preprocessing
    section("Preprocessing Pipeline")
    pre_stats = bench_preprocessing(args.image, args.runs)
    results["preprocessing"] = pre_stats
    row("Source",  pre_stats["source"])
    row("Min",     pre_stats["min_ms"],    "ms")
    row("Mean",    pre_stats["mean_ms"],   "ms")
    row("Median",  pre_stats["median_ms"], "ms")
    row("Max",     pre_stats["max_ms"],    "ms")

    # End-to-end
    section("End-to-End Pipeline  (preproc + infer + argmax)")
    e2e = bench_end_to_end(interp, image_batch, args.runs)
    results["end_to_end"] = e2e
    row("Min",        e2e["min_ms"],         "ms")
    row("Mean",       e2e["mean_ms"],         "ms")
    row("Median",     e2e["median_ms"],       "ms")
    row("P95",        e2e["p95_ms"],          "ms")
    row("Throughput", e2e["throughput_fps"],  "FPS")

    # Memory
    section("Memory Usage")
    mem = bench_memory(interp, image_batch)
    results["memory"] = mem
    if "note" not in mem:
        row("RAM before inference", mem["before_mb"], "MB")
        row("RAM after inference",  mem["after_mb"],  "MB")
        row("Delta",                mem["delta_mb"],  "MB")
    else:
        row("Note", mem["note"])

    # CPU usage
    section(f"CPU Usage  ({args.runs} runs burst)")
    cpu = bench_cpu_usage(interp, image_batch, args.runs)
    results["cpu_usage"] = cpu
    if "note" not in cpu:
        row("CPU utilisation",  cpu["cpu_percent"],  "%")
        row("Total burst time", cpu["total_time_s"], "s")
    else:
        row("Note", cpu["note"])

    # Summary + verdict
    section("Summary")
    mean_inf  = inf_stats["mean_ms"]
    mean_pre  = pre_stats["mean_ms"]
    total_est = round(mean_pre + mean_inf, 2)
    row("Avg preprocessing",     mean_pre,   "ms")
    row("Avg inference",         mean_inf,   "ms")
    row("Estimated total/image", total_est,  "ms")
    row("CPU temp (Pi only)",    sys_info.get("cpu_temp", "N/A"))
    row("Runtime backend",       RUNTIME)
    print()

    if mean_inf < 300:
        verdict = "EXCELLENT  (<300 ms)  -- real-time capable"
    elif mean_inf < 600:
        verdict = "GOOD       (<600 ms)  -- suitable for deployment"
    elif mean_inf < 1200:
        verdict = "ACCEPTABLE (<1.2 s)   -- usable but consider optimisation"
    else:
        verdict = "SLOW       (>1.2 s)   -- consider quantisation or model pruning"
    row("Verdict", verdict)

    # Save JSON
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["config"] = {
        "runs": args.runs, "threads": args.threads,
        "image": args.image or "synthetic"
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
