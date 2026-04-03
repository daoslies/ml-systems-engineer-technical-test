My slightly messy dev changelog is in changelog.md - this is a neater version for your perusal.


# Section 4 — Inference Server & Dockerfile: Changelog (Structured by Severity/Category)

## 1. Model Reloaded on Every Request (Correctness, Performance)
- **Issue:** The model was loaded and moved to GPU inside the `/predict` endpoint for every request.
- **Why it matters:** This caused huge latency, wasted GPU memory, and could lead to OOM errors under load.
- **Fix:** Refactored to load the model once at server startup and reuse it for all requests (class-based structure).

## 2. No Batch Processing (Performance, GPU Hygiene)
- **Issue:** Each frame was processed individually, missing out on GPU batch parallelism.
- **Why it matters:** This limited throughput and increased per-frame overhead.
- **Fix:** Batched all frames into a single tensor and ran them through the model in one forward pass.

## 3. Inefficient Preprocessing (Performance)
- **Issue:** Used per-frame torch transforms and device transfers.
- **Why it matters:** Increased CPU and GPU overhead, especially for small batches.
- **Fix:** Moved all preprocessing to fast, batched NumPy/OpenCV operations on CPU, then transferred the batch to GPU in one go.

## 4. No torch.no_grad() or inference_mode() (Correctness, Performance)
- **Issue:** Inference was run without disabling autograd.
- **Why it matters:** Unnecessary memory usage and slower inference.
- **Fix:** Wrapped inference in `torch.inference_mode()` for optimal performance and memory usage.

## 5. Device Transfer and Synchronization Issues (GPU Hygiene)
- **Issue:** Unnecessary or misplaced `.cuda()` calls and `torch.cuda.synchronize()` calls.
- **Why it matters:** Wasted time, risked deadlocks, and could cause subtle bugs.
- **Fix:** Ensured all device transfers are done once, at the right time, and removed unnecessary synchronizations.

## 6. Poor Output Encoding (Performance)
- **Issue:** Used PIL/PNG for output frame encoding.
- **Why it matters:** PNG encoding is slow; PIL is less efficient for real-time workloads.
- **Fix:** Switched to OpenCV/JPEG encoding, significantly reducing postprocessing latency.

## 7. Flask + Gunicorn Instead of FastAPI + Uvicorn (Production Readiness)
- **Issue:** Used Flask, which is less performant and scalable for async workloads.
- **Why it matters:** Limited concurrency and scalability.
- **Fix:** Migrated to FastAPI + Uvicorn for modern, async-native, high-performance serving.

## 8. Dockerfile: Pinning, Lean Runtime, Model Weights, and Layer Caching (Container Best Practices)
- **Issue:** Previous versions used multi-stage builds and sometimes cached model weights for the wrong user, leading to runtime downloads. Also, dev tools were sometimes present in the final image. Non-root user was attempted but removed due to model weight cache path conflicts between build and runtime.
- **Why it matters:** Multi-stage was unnecessary for this use case, and incorrect weight caching led to runtime delays. Dev tools increase image size and attack surface. Non-root is best practice, but in this case, would require setting TORCH_HOME or similar to unify cache paths.
- **Fix:** 
  - Switched to a single-stage build with a pinned base image.
  - Only runtime dependencies are installed.
  - Model weights are downloaded at build time for the runtime user, avoiding runtime downloads. (Non-root user would be used in production with explicit cache path.)
  - Added HEALTHCHECK and tini for signal handling.
  - CMD runs uvicorn with a single worker.
  - All Python dependencies are in requirements.txt for reproducibility.
  - COPY requirements.txt . then COPY . . ordering ensures Docker layer caching is optimal: requirements changes won't invalidate the code layer, speeding up rebuilds.

---

**Other Notable Improvements:**
- Improved input validation (batch size, dimensions).
- Preserved and improved logging/timing for benchmarking.
- Added comments and structure for maintainability.

---

# Mapping to Original Changelog
- Each of the above points is supported by detailed benchmarking and experimentation, as documented in the original changelog (see [Change 1]–[Change 13]).
- All reverted experiments (parallelization, pin_memory, etc.) are noted for transparency and completeness.
- Container and code best practices are explicitly listed for clarity.

---

This structured changelog is provided alongside the original for clarity and completeness.
