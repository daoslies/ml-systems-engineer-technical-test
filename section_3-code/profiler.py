"""
profiler.py — Signapse ML Systems Engineer Assessment
======================================================
Benchmarks a PyTorch model (resnet18 by default) and reports latency
statistics, bottleneck classification, and optionally an ONNX comparison.

Usage:
    python profiler.py
    python profiler.py --warmup 20 --runs 200 --batch-size 4
    python profiler.py --onnx --device cpu

Requirements:
    torch torchvision numpy
    Optional: pynvml (GPU utilisation), onnxruntime (ONNX benchmark)
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import onnxruntime as ort
    print("\n[ONNX] onnxruntime imported successfully.")
    ort.preload_dlls() 
    print("\n[ONNX] onnxruntime DLLs preloaded successfully.")
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:

    label: str
    latencies_ms: list[float]
    cpu_samples: list[float] = field(default_factory=list)
    gpu_samples: list[float] = field(default_factory=list)
    outlier_indices: list[int] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        """Throughput in samples/sec."""
        total_time = sum(self.latencies_ms)
        if total_time == 0:
            return 0.0
        return 1000.0 * len(self.latencies_ms) / total_time

    @property
    def p50(self) -> float:
        return float(np.percentile(self.latencies_ms, 50))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def p99(self) -> float:
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def mean(self) -> float:
        return float(np.mean(self.latencies_ms))

    @property
    def std(self) -> float:
        return float(np.std(self.latencies_ms))

    @property
    def mean_cpu_util(self) -> Optional[float]:
        return float(np.mean(self.cpu_samples)) if self.cpu_samples else None

    @property
    def mean_gpu_util(self) -> Optional[float]:
        return float(np.mean(self.gpu_samples)) if self.gpu_samples else None

    def classify_bottleneck(self) -> str:
        """
        Classify whether the workload is GPU-bound, CPU-bound, or balanced.

        Heuristic:
          - If GPU util > 80% and CPU util < 60%  → GPU-bound
          - If CPU util > 80% and GPU util < 60%  → CPU-bound
          - If both > 60%                          → balanced / pipeline-bound
          - If no GPU available                    → CPU-bound by definition
          - If utilisation data unavailable        → inconclusive
        """
        cpu = self.mean_cpu_util
        gpu = self.mean_gpu_util

        if gpu is None and cpu is None:
            return "inconclusive (no utilisation data)"

        if gpu is None:
            # CPU-only machine
            if cpu is not None and cpu > 70:
                return "CPU-bound"
            return "CPU-bound (no GPU detected)"

        if cpu is not None:
            if gpu > 80 and cpu < 60:
                return "GPU-bound"
            if cpu > 80 and gpu < 60:
                return "CPU-bound"
            if gpu > 60 and cpu > 60:
                return "balanced / pipeline-bound"
            # Both low — likely memory or I/O bottleneck
            return "memory / I/O-bound (low CPU and GPU utilisation)"

        return "inconclusive"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latency profiler for PyTorch inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warm-up passes (not timed)")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of timed inference passes")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per inference pass")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or None to auto-detect")
    parser.add_argument("--onnx", action="store_true",
                        help="(Bonus) Export to ONNX and run side-by-side benchmark")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Spatial size of the input tensor (square)")
    parser.add_argument("--csv", type=str, default=None, 
                        help="Write summary results to this CSV file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model + input
# ---------------------------------------------------------------------------

def load_model(device: torch.device) -> torch.nn.Module:
    """Load resnet18 in eval mode on the target device."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model


def make_input(batch_size: int, input_size: int, device: torch.device) -> torch.Tensor:
    """Construct a random input tensor matching resnet's expected format."""
    return torch.randn(batch_size, 3, input_size, input_size, device=device)


# ---------------------------------------------------------------------------
# Utilisation sampling
# ---------------------------------------------------------------------------

def _gpu_utilisation_pynvml() -> Optional[float]:
    """Sample GPU utilisation % via pynvml (most reliable method)."""
    if not PYNVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(rates.gpu)
    except Exception:
        return None


def _gpu_utilisation_smi() -> Optional[float]:
    """
    Fallback: parse nvidia-smi output.
    Slower than pynvml — subprocess overhead per call — so only used
    when pynvml is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def sample_utilisation() -> tuple[Optional[float], Optional[float]]:
    """
    Returns (cpu_percent, gpu_percent).
    Either may be None if the data source is unavailable.
    """
    cpu = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else None

    gpu = _gpu_utilisation_pynvml()
    if gpu is None:
        gpu = _gpu_utilisation_smi()

    return cpu, gpu


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    label: str,
    infer_fn,
    input_tensor: torch.Tensor,
    warmup: int,
    runs: int,
    device: torch.device,
) -> BenchmarkResult:
    """
    Run warm-up passes (untimed) then timed passes.
    """
    use_cuda = device.type == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    print(f"\n[{label}] Warming up ({warmup} passes)...", end=" ", flush=True)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = infer_fn(input_tensor)
    sync()
    print("done.")

    # Prime utilisation sampler — first psutil call always returns 0.0
    if PSUTIL_AVAILABLE:
        psutil.cpu_percent(interval=None)

    latencies: list[float] = []
    cpu_samples: list[float] = []
    gpu_samples: list[float] = []

    print(f"[{label}] Timing {runs} passes...", end=" ", flush=True)
    print("\n=== BENCHMARK TIMING START ===", flush=True)
    with torch.inference_mode():
        for i in range(runs):
            # Sample before every inference
            cpu_pre, gpu_pre = sample_utilisation()

            sync()
            t0 = time.perf_counter()
            _ = infer_fn(input_tensor)
            sync()
            t1 = time.perf_counter()

            # Sample after every inference
            cpu_post, gpu_post = sample_utilisation()

            latencies.append((t1 - t0) * 1000)  # convert to ms

            # Average pre/post samples for a more stable reading
            if cpu_pre is not None and cpu_post is not None:
                cpu_samples.append((cpu_pre + cpu_post) / 2)
            if gpu_pre is not None and gpu_post is not None:
                gpu_samples.append((gpu_pre + gpu_post) / 2)
    print("=== BENCHMARK TIMING END ===", flush=True)
    print("done.")

    # Flag outliers: passes > mean + 2*std
    arr = np.array(latencies)
    threshold = arr.mean() + 2 * arr.std()
    outlier_indices = [i for i, v in enumerate(latencies) if v > threshold]

    return BenchmarkResult(
        label=label,
        latencies_ms=latencies,
        cpu_samples=cpu_samples,
        gpu_samples=gpu_samples,
        outlier_indices=outlier_indices,
    )


# ---------------------------------------------------------------------------
# ONNX export + benchmark (bonus)
# ---------------------------------------------------------------------------

def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, path: Path) -> bool:
    """
    Export the model to ONNX with dynamic batch size.
    Returns True on success.
    """
    if not ONNXRUNTIME_AVAILABLE:
        print("\n[ONNX] onnxruntime not installed — skipping. Install with: pip install onnxruntime")
        return False

    print(f"\n[ONNX] Exporting model to {path}...", end=" ", flush=True)
    try:
        # Move to CPU for export — TorchScript/ONNX export works more
        # reliably from CPU; the resulting ONNX graph is device-agnostic.
        cpu_input = dummy_input.cpu()
        cpu_model = model.cpu()

        torch.onnx.export(
            cpu_model,
            cpu_input,
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=18,
        )
        print("done.")
        return True
    except Exception as e:
        print(f"\n[ONNX] Export failed: {e}")
        return False


def make_onnx_infer_fn(onnx_path: Path, input_size: int, batch_size: int):
    """Return a callable that runs one ONNX inference pass via onnxruntime."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    def get_device():
        # Try to infer device from available providers
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "cuda"
        return "cpu"

    device = get_device()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
    )
    input_name = session.get_inputs()[0].name

    print(f"\n[ONNX] ------------------------------------ \n")
    print(f"[ONNX] Providers: {session.get_providers()}")
    print(f"[ONNX] Provider options: {session.get_provider_options()}")
    if "CUDAExecutionProvider" not in session.get_providers():
        print("[ONNX WARNING] CUDA provider not active — falling back to CPU")

    if device == "cuda":
        io_binding = session.io_binding()
        def infer(tensor: torch.Tensor):
            np_input = tensor.detach().cpu().numpy()
            io_binding.bind_cpu_input(input_name, np_input)
            io_binding.bind_output("output")
            session.run_with_iobinding(io_binding)
            return io_binding.copy_outputs_to_cpu()
    else:
        def infer(tensor: torch.Tensor):
            np_input = tensor.cpu().numpy()
            return session.run(None, {input_name: np_input})

    return infer


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

COL_W = 14  # column width for the summary table

def _row(*cells) -> str:
    return "  ".join(str(c).rjust(COL_W) for c in cells)

def _divider(n_cols: int) -> str:
    return "-" * (COL_W * n_cols + 2 * (n_cols - 1))


def print_single_table(result: BenchmarkResult) -> None:
    headers = ["metric", "value"]
    rows = [
        ("p50 (ms)",   f"{result.p50:.2f}"),
        ("p95 (ms)",   f"{result.p95:.2f}"),
        ("p99 (ms)",   f"{result.p99:.2f}"),
        ("mean (ms)",  f"{result.mean:.2f}"),
        ("std (ms)",   f"{result.std:.2f}"),
        ("throughput (samples/sec)", f"{result.throughput:.2f}"),
        ("runs",       str(len(result.latencies_ms))),
        ("outliers",   f"{len(result.outlier_indices)} ({100*len(result.outlier_indices)/len(result.latencies_ms):.1f}%)"),
    ]
    if result.mean_cpu_util is not None:
        rows.append(("cpu util",  f"{result.mean_cpu_util:.1f}%"))
    if result.mean_gpu_util is not None:
        rows.append(("gpu util",  f"{result.mean_gpu_util:.1f}%"))
    rows.append(("bottleneck", result.classify_bottleneck()))

    print(f"\n{'─'*50}")
    print(f"  {result.label}")
    print(f"{'─'*50}")
    for label, value in rows:
        print(f"  {label:<20}{value}")
    print(f"{'─'*50}")


def print_comparison_table(pytorch: BenchmarkResult, onnx: BenchmarkResult) -> None:
    """Side-by-side comparison table for PyTorch vs ONNX."""
    col_labels = ["metric", "pytorch", "onnx", "delta"]
    divider = "─" * (COL_W * 4 + 6)

    def speedup(a, b):
        # positive = ONNX is faster, negative = ONNX is slower
        if a == 0:
            return "n/a"
        pct = (a - b) / a * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.1f}%"

    metrics = [
        ("p50 (ms)",  pytorch.p50,  onnx.p50),
        ("p95 (ms)",  pytorch.p95,  onnx.p95),
        ("p99 (ms)",  pytorch.p99,  onnx.p99),
        ("mean (ms)", pytorch.mean, onnx.mean),
        ("std (ms)",  pytorch.std,  onnx.std),
        ("throughput (samples/sec)", pytorch.throughput, onnx.throughput),
    ]

    print(f"\n{'─'*60}")
    print(f"  PyTorch vs ONNX — side-by-side")
    print(f"{'─'*60}")
    print(f"  {'metric':<18}{'pytorch':>10}  {'onnx':>10}  {'delta (onnx vs pytorch)':>24}")
    print(f"{'─'*60}")
    for label, pt_val, onnx_val in metrics:
        print(f"  {label:<18}{pt_val:>10.2f}  {onnx_val:>10.2f}  {speedup(pt_val, onnx_val):>24}")
    print(f"{'─'*60}")

    pt_bot = pytorch.classify_bottleneck()
    onnx_bot = onnx.classify_bottleneck()
    print(f"  {'bottleneck':<18}{'pytorch: ' + pt_bot}")
    print(f"  {'':18}{'onnx:    ' + onnx_bot}")
    print(f"{'─'*60}")

    # Verdict
    speedup_mean = (pytorch.mean - onnx.mean) / pytorch.mean * 100 if pytorch.mean > 0 else 0
    if speedup_mean > 5:
        verdict = f"ONNX is {speedup_mean:.1f}% faster on mean latency — worth investigating further."
    elif speedup_mean < -5:
        verdict = f"ONNX is {abs(speedup_mean):.1f}% slower — straight PyTorch is preferable for this workload."
    else:
        verdict = "Negligible difference (<5%) — ONNX conversion adds complexity without meaningful gain here."
    print(f"\n  Verdict: {verdict}")
    print(f"{'─'*60}")


def print_outlier_report(result: BenchmarkResult) -> None:
    if not result.outlier_indices:
        return
    threshold = result.mean + 2 * result.std
    print(f"\n  Outlier passes (>{threshold:.2f}ms, i.e. mean + 2σ):")
    for idx in result.outlier_indices:
        print(f"    pass {idx+1:>4d}: {result.latencies_ms[idx]:.2f}ms")



import csv
import os

def write_csv_row(csv_path, result, args):
    header = [
        "label", "device", "batch_size", "input_size", "warmup", "runs",
        "p50", "p95", "p99", "mean", "std", "cpu_util", "gpu_util", "outliers"
    ]
    row = [
        result.label,
        str(args.device),
        args.batch_size,
        args.input_size,
        args.warmup,
        args.runs,
        f"{result.p50:.4f}",
        f"{result.p95:.4f}",
        f"{result.p99:.4f}",
        f"{result.mean:.4f}",
        f"{result.std:.4f}",
        f"{result.mean_cpu_util if result.mean_cpu_util is not None else ''}",
        f"{result.mean_gpu_util if result.mean_gpu_util is not None else ''}",
        f"{len(result.outlier_indices)}"
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"        ---  Signapse inference profiler   ---")
    print(f"{'='*60}")
    print(f"  device      : {device}")
    print(f"  model       : resnet18")
    print(f"  input size  : {args.batch_size} × 3 × {args.input_size} × {args.input_size}")
    print(f"  warmup runs : {args.warmup}")
    print(f"  timed runs  : {args.runs}")
    print(f"  pynvml      : {'available' if PYNVML_AVAILABLE else 'not available'}")
    print(f"  psutil      : {'available' if PSUTIL_AVAILABLE else 'not available'}")
    print(f"  onnxruntime : {'available' if ONNXRUNTIME_AVAILABLE else 'not available'}")
    print(f"{'='*60}")

    # Load model
    model = load_model(device)
    dummy_input = make_input(args.batch_size, args.input_size, device)

    # PyTorch benchmark
    pytorch_result = run_benchmark(
        label="PyTorch",
        infer_fn=model,
        input_tensor=dummy_input,
        warmup=args.warmup,
        runs=args.runs,
        device=device,
    )

    # ONNX benchmark (bonus)
    onnx_result = None
    if args.onnx:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)

        exported = export_to_onnx(model, dummy_input, onnx_path)
        if exported:
            onnx_infer = make_onnx_infer_fn(onnx_path, args.input_size, args.batch_size)
            # ONNX runs on CPU input; keep the dummy on CPU for fair comparison
            #cpu_dummy = dummy_input.cpu()
            onnx_result = run_benchmark(
                label="ONNX",
                infer_fn=onnx_infer,
                input_tensor=dummy_input,
                warmup=args.warmup,
                runs=args.runs,
                device=torch.device("cpu"),  # utilisation tracked on CPU side
            )
        try:
            onnx_path.unlink()
        except Exception:
            pass

        # Write results to CSV if requested
    if args.csv:
        if onnx_result:
            write_csv_row(args.csv, pytorch_result, args)
            write_csv_row(args.csv, onnx_result, args)
        else:
            write_csv_row(args.csv, pytorch_result, args)

    # Print results
    if onnx_result:
        print_comparison_table(pytorch_result, onnx_result)
        print_outlier_report(pytorch_result)
        print_outlier_report(onnx_result)
    else:
        print_single_table(pytorch_result)
        print_outlier_report(pytorch_result)

    print()


if __name__ == "__main__":
    main()
