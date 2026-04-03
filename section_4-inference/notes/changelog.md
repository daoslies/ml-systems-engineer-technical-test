## [Change 13]

**Description:** Refactored Dockerfile for production best practices and reproducibility. Implemented a multi-stage build, pinned the base image, installed dev/build tools only in the builder stage, removed dev tools from the final image, added a non-root user, included a HEALTHCHECK, used tini for signal handling, and set CMD to run uvicorn with a single worker. Moved all Python dependencies to requirements.txt and created the requirements.txt file for clarity and reproducibility.

**Reason:** These changes improve security, reduce image size, ensure proper signal handling, enable health monitoring, and make dependency management more maintainable and reproducible.

**Implementation Plan:**
- Updated Dockerfile to use multi-stage build and best practices.
- Created requirements.txt and moved all Python dependencies there.
- Added non-root user and HEALTHCHECK.
- Used tini as entrypoint for signal handling.
- CMD runs uvicorn with a single worker.
## [Change 12]

Removed torch.cuda.synchronize() after moving to inference mode

## [Change 11]

Moved to inference mode from no_grad



## [Change 10]
**Description:** Experimented with moving the .cuda() device transfer step to different points in the preprocessing pipeline, and with parallelizing the OpenCV resize step using ThreadPoolExecutor. Both changes were benchmarked and reverted after observing no speedup (and sometimes a slowdown) for our current workload.

**Reason:** Theoretical optimizations like parallelizing CPU-bound steps or batching device transfers can help for large batches, but for our small batch size and current data, the overhead outweighed any benefit. Sequential preprocessing and a single .cuda() after all CPU work remains optimal for now.

**Benchmark Plan:**
- Moved .cuda() to different locations in preprocessing and benchmarked ('Editing preprocess frames - moving around the cuda - v2').
- Parallelized cv2.resize with ThreadPoolExecutor and benchmarked ('Editing preprocess frames threading').
- Both changes reverted after confirming no improvement in benchmark_history.csv.


## [Change 9]
**Description:** Moved all preprocessing away from torch transforms to fully batched, vectorized NumPy/OpenCV operations. Timing variable t4 is now set to t3 after interpolation to maintain logging/benchmark compatibility.

**Reason:** Batched NumPy/OpenCV preprocessing is much faster than per-frame torch transforms, especially for small batch sizes. This change yielded a measurable speedup in preprocessing and overall latency. Setting t4 = t3 ensures the logging/timing structure remains compatible with previous benchmarks and analysis scripts.

**Benchmark Plan:**
- Implemented as a single commit ('Moved away from torch for preprocessing (t4 now = t3)').
- Benchmarked and observed a clear speedup in preprocessing and overall latency (see benchmark_history.csv).
- Logging/timing structure is preserved for downstream analysis.


## [Change 8]
**Description:** Attempted to parallelize postprocessing with ThreadPoolExecutor for multi-frame requests, then reverted.

**Reason:** Parallelization can speed up CPU-bound tasks for large batches, but for batch size 1 it added overhead and did not improve speed.

**Result:** Benchmarked ('parrellelise post processing') and observed no improvement for current workload; reverted in 'REVERT parrellelise post processing'.
## [Change 7]
**Description:** Switched postprocessing from PIL/PNG to OpenCV/JPEG encoding for output frames.

**Reason:** OpenCV JPEG encoding is significantly faster than PIL PNG, especially for real-time inference workloads. This reduces postprocessing latency and increases throughput.

**Benchmark Plan:**
- Implemented as a single commit ('pil to opencv | png to jpeg').
- Benchmarked and observed a clear speedup in postprocessing and overall latency (see benchmark_history.csv).
## [Change 6]
**Description:** Experimented with moving all preprocessing to CPU, using .pin_memory() in preprocess_frame, and moving the batch to CUDA in one go with .cuda(non_blocking=True).

**Reason:** This is a common optimization for large batch inference workloads, as it can reduce CPU-GPU transfer overhead and improve throughput when the transfer is a bottleneck.

**Result:** For our current workload (batch size 1, small images), this change did not yield a speedup and sometimes slightly increased latency. We reverted to the previous approach of moving each frame to CUDA after preprocessing.

**Benchmark Plan:**
- Implemented as a single commit ('use pin memory insetad of .cuda').
- Benchmarked and compared to previous best. No improvement observed, so reverted.
## [Change 5]
- Added torch.no_grad() context to inference, reducing latency further (see benchmark_history.csv, commit 'Use torch no grad').

## [Change 4]
**Description:** Migrate inference server from Flask to FastAPI + Uvicorn for improved concurrency, async support, and modern API design.

**Reason:** FastAPI + Uvicorn is more performant for high-concurrency workloads, is async-native, and is easier to scale and maintain. This migration will allow us to handle more requests efficiently and future-proof the server.

**Migration Plan:**
- Install FastAPI and Uvicorn in Dockerfile.
- Refactor Flask app to FastAPI, preserving all endpoint logic and timing.
- Update Dockerfile CMD to use Uvicorn.
- Update run scripts and health checks as needed.
- Benchmark and compare performance before/after migration.


- Migrated inference server from Flask to FastAPI/uvicorn.
- Achieved significant speedup in /predict endpoint latency (see benchmark_history.csv, commit 'migration to fastapi').


## [Change 3]
**Description:** Batch frame processing. Instead of processing each frame individually, stack all frames into a single batch tensor and run them through the model in one forward pass.

**Reason:** Leverages GPU parallelism, reduces per-frame overhead, and should significantly improve throughput for multi-frame requests.

**Benchmark Plan:**
- Implement this change as a single commit.
- Run the benchmark and record the result with a descriptive commit message.



---

## [Change 1]
**Description:** Refactored the inference server to use a class-based structure. The model is now loaded once at startup in `self.model` (via a method), and all logic is encapsulated in the `InferenceServer` class. The model is reused for all requests.

**Reason:** This structure is more maintainable and extensible, and ensures the model is not reloaded on every request, which should significantly reduce per-request latency and improve throughput.

**Benchmark Plan:**
- Implemented as a single commit.
- Run the benchmark and recorded the result with a descriptive commit message.

---




----


Put in clearer line names on the graph. specifically which is inference server, and which is return speed.

Timestamps: remove year and 45 degree angle. 

Work out what the difference is between inference server speed, and api return speed. 

split out the subsection timings into their own graph.


# Key fixes for app.py
- Lazy CUDA init (not in __init__)
- Input validation (max batch size, dimension checks)
- torch.no_grad() wrapper for inference
- Move preprocessing to CPU, pin_memory for async transfer
- Add @torch.inference_mode() (PyTorch 1.9+)
- Replace f-string logging with % formatting
- Add semaphore to limit concurrent GPU requests (even with 1 worker)


# Key fixes for Dockerfile
- Pin base image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
- Use multi-stage build to reduce final image size
- Install dev/build tools only in builder stage
- Remove dev tools from final image (only runtime deps remain)
- Add non-root user (appuser) and run as non-root
- Add HEALTHCHECK for container health
- Use tini for signal handling and proper shutdown
- CMD with single uvicorn worker (or gunicorn+gevent if needed)
- Move Python dependencies to requirements.txt for reproducibility