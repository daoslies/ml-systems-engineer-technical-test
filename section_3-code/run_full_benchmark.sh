#!/bin/bash
# Run a full suite of profiling tests with profiler.py and collect results in a CSV

set -e




# ========== Configurable HYPERPARAMETERS ========== #


WARMUP=50
RUNS=100
BATCH_SIZES=(1 16)




cd "$(dirname "$0")"

CSV_OUT="benchmark_results.csv"
rm -f "$CSV_OUT"


# CPU only
for BS in "${BATCH_SIZES[@]}"; do
    python3 profiler.py --warmup $WARMUP --runs $RUNS --batch-size $BS --device cpu --csv "$CSV_OUT"
done

# GPU (if available)
if python3 -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    for BS in "${BATCH_SIZES[@]}"; do
        python3 profiler.py --warmup $WARMUP --runs $RUNS --batch-size $BS --device cuda --csv "$CSV_OUT"
    done
else
    echo "[INFO] CUDA not available, skipping GPU tests."
fi

# ONNX (CPU)
for BS in "${BATCH_SIZES[@]}"; do
    python3 profiler.py --onnx --warmup $WARMUP --runs $RUNS --batch-size $BS --device cpu --csv "$CSV_OUT"
done

# ONNX (GPU, if available)
if python3 -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    for BS in "${BATCH_SIZES[@]}"; do
        python3 profiler.py --onnx --warmup $WARMUP --runs $RUNS --batch-size $BS --device cuda --csv "$CSV_OUT"
    done
else
    echo "[INFO] CUDA not available, skipping ONNX GPU tests."
fi

echo "All results written to $CSV_OUT"

python3 plot_results.py

echo "Plots generated from $CSV_OUT"