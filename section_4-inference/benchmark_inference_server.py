"""
Benchmark script for section_4-inference Flask server.
Sends POST requests to /predict and measures latency, throughput, and outputs results as CSV and table.
"""
import argparse
import time
import requests
import numpy as np
import csv
import sys
import os


def generate_dummy_frame(size=224):
    # Generate a dummy pose tensor (flattened list)
    return np.random.rand(size * size).tolist()


def benchmark_predict(

    url,
    num_requests=100,
    frames_per_request=1,
    size=224,
    warmup=10,
    csv_out=None,
    commit_message=None,
):
    latencies = []
    timing_fields = [
        "preprocessing", "batching", "resize", "cuda_sync_before", "inference", "postprocessing", "total",
        # New GPU timing fields
        "gpu_idle_before", "gpu_active", "launch_overhead"
    ]
    timing_sums = {k: 0.0 for k in timing_fields}
    timing_counts = 0
    for i in range(num_requests + warmup):
        payload = {
            "frames": [generate_dummy_frame(size) for _ in range(frames_per_request)],
            "size": size,
        }
        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            resp_json = resp.json()
            if i < 3:  # Print the first 3 responses for debugging
                #print(f"/predict response {i+1}: {resp_json}")
                pass
            timing = None
            try:
                timing = resp_json.get("timing")
            except Exception:
                timing = None
        except Exception as e:
            print(f"Request {i+1} failed: {e}", file=sys.stderr)
            continue
        elapsed = (time.time() - start) * 1000  # ms
        if i >= warmup:
            latencies.append(elapsed)
            if timing:
                for k in timing_fields:
                    if k in timing:
                        timing_sums[k] += timing[k]
                timing_counts += 1
        if (i+1) % 10 == 0:
            print(f"Completed {i+1} / {num_requests + warmup} requests...")

    latencies = np.array(latencies)
    print("\nResults:")
    print(f"  Requests: {len(latencies)}")
    if len(latencies) == 0:
        print("  No successful requests. Check if the server is running and reachable.")
        p50 = p95 = p99 = mean = std = throughput = 'NaN'
        timing_means = {k: 'NaN' for k in timing_fields}
    else:
        p50 = f"{np.percentile(latencies, 50):.2f}"
        p95 = f"{np.percentile(latencies, 95):.2f}"
        p99 = f"{np.percentile(latencies, 99):.2f}"
        mean = f"{latencies.mean():.2f}"
        std = f"{latencies.std():.2f}"
        throughput = f"{1000 * len(latencies) / latencies.sum():.2f}"
        print(f"  p50: {p50} ms")
        print(f"  p95: {p95} ms")
        print(f"  p99: {p99} ms")
        print(f"  Mean: {mean} ms")
        print(f"  Std: {std} ms")
        print(f"  Throughput: {throughput} req/s")
        if timing_counts > 0:
            timing_means = {k: f"{(timing_sums[k]/timing_counts):.6f}" for k in timing_fields}
            print("  Timing breakdown (mean, seconds):")
            # Group for readability
            existing = ["preprocessing", "batching", "resize", "cuda_sync_before",
                        "inference", "postprocessing", "total"]
            gpu_new  = ["gpu_idle_before", "gpu_active", "launch_overhead"]
            for k in existing:
                print(f"    {k:20s}: {timing_means[k]}")
            print("  GPU Event timing (mean, seconds):")
            for k in gpu_new:
                print(f"    {k:20s}: {timing_means[k]}")
        else:
            timing_means = {k: 'NaN' for k in timing_fields}

    if csv_out:
        header = [
            "timestamp", "commit_message", "url", "num_requests", "frames_per_request", "size",
            "p50_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms", "throughput_rps"
        ] + [f"timing_{k}_s" for k in timing_fields]
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            commit_message,
            url,
            num_requests,
            frames_per_request,
            size,
            p50,
            p95,
            p99,
            mean,
            std,
            throughput
        ] + [timing_means[k] for k in timing_fields]
        # Check if header matches, and rewrite if not
        need_header = False
        if not os.path.exists(csv_out):
            need_header = True
        else:
            with open(csv_out, "r", newline="") as f:
                first_line = f.readline().strip()
                current_header = [h.strip() for h in first_line.split(",")]
                if current_header != header:
                    # Rewrite the file with the new header and preserve old rows
                    old_rows = f.readlines()
                    with open(csv_out, "w", newline="") as fw:
                        writer = csv.writer(fw)
                        writer.writerow(header)
                        for old_row in old_rows:
                            fw.write(old_row)
                    need_header = False  # Already written
        with open(csv_out, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(header)
            writer.writerow(row)
        print(f"Results appended to {csv_out}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark section_4-inference server.")
    parser.add_argument("--url", type=str, default="http://localhost:8080/predict", help="/predict endpoint URL")
    parser.add_argument("--num-requests", type=int, default=50, help="Number of timed requests")
    parser.add_argument("--frames-per-request", type=int, default=1, help="Frames per request")
    parser.add_argument("--size", type=int, default=224, help="Frame size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup requests (not timed)")
    parser.add_argument("--csv", type=str, default="benchmark_history.csv", help="CSV output file (never reset)")
    parser.add_argument("--commit-message", type=str, default=None, help="Commit message describing this run (required)")
    args = parser.parse_args()

    if not args.commit_message:
        print("Error: --commit-message is required for benchmarking runs.", file=sys.stderr)
        sys.exit(1)

    benchmark_predict(
        url=args.url,
        num_requests=args.num_requests,
        frames_per_request=args.frames_per_request,
        size=args.size,
        warmup=args.warmup,
        csv_out=args.csv,
        commit_message=args.commit_message,
    )

if __name__ == "__main__":
    main()
