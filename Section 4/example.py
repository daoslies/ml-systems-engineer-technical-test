"""
Signapse — ML Systems Engineer Technical Assessment
Section 4: Inference Latency Profiler

Starter scaffold. You are not required to use this — feel free to start from scratch.
"""

import time
import numpy as np
import torch
import torchvision.models as models

# ── Configuration ────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_PASSES = 10
TIMED_PASSES = 100
INPUT_SHAPE = (1, 3, 224, 224)  # Batch of 1, RGB, 224x224


# ── Model setup ──────────────────────────────────────────────────────────────

def load_model():
    """Load and prepare the model for inference."""
    model = models.resnet18(weights=None)
    model.eval()
    model.to(DEVICE)
    return model


# ── Benchmarking ─────────────────────────────────────────────────────────────

def run_benchmark(model, warmup: int = WARMUP_PASSES, timed: int = TIMED_PASSES):
    """
    Run warmup passes then timed passes.
    Returns a list of per-pass latencies in milliseconds.
    """
    dummy_input = torch.randn(INPUT_SHAPE).to(DEVICE)
    latencies = []

    # TODO: run warmup passes

    # TODO: run timed passes, recording latency for each
    # Hint: use torch.cuda.synchronize() before/after each pass if on GPU
    # to ensure accurate timing

    return latencies


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(latencies: list[float]) -> dict:
    """Compute summary statistics from a list of latencies (ms)."""
    arr = np.array(latencies)
    return {
        "mean_ms": None,    # TODO
        "std_ms": None,     # TODO
        "p50_ms": None,     # TODO
        "p95_ms": None,     # TODO
        "p99_ms": None,     # TODO
    }


# ── Utilisation ──────────────────────────────────────────────────────────────

def get_utilisation() -> dict:
    """
    Return a snapshot of CPU and (optionally) GPU utilisation.
    Use any method you like: psutil, pynvml, subprocess nvidia-smi, etc.
    """
    return {
        "cpu_percent": None,   # TODO
        "gpu_percent": None,   # TODO (None if not available)
        "gpu_memory_used_mb": None,  # TODO (None if not available)
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def print_summary(stats: dict, utilisation: dict):
    """Print a readable summary table to stdout."""
    # TODO: implement
    pass


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    model = load_model()
    latencies = run_benchmark(model)
    stats = compute_stats(latencies)
    utilisation = get_utilisation()
    print_summary(stats, utilisation)