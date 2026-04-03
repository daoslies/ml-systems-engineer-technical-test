


🔴 CRITICAL


1. Hardcoded API key in environment

    Issue: Secret embedded directly in manifest

    Impact: Credential leakage via source control, logs, and cluster introspection

    Fix: Moved to Kubernetes Secret using valueFrom.secretKeyRef

2. Deployment strategy set to Recreate

    Issue: All pods terminated before new ones are created

    Impact: Guaranteed downtime during deployments, especially problematic for ML workloads with cold start times

    Fix: Replaced with RollingUpdate strategy with zero downtime configuration

3. terminationGracePeriodSeconds: 0

    Issue: Immediate container termination (SIGKILL)

    Impact: Drops in-flight inference requests, causing user-visible errors

    Fix: Increased to 60 seconds and added preStop hook for graceful draining

4. Missing GPU resource requests

    Issue: No nvidia.com/gpu specified

    Impact: Pod may be scheduled on non-GPU nodes or experience undefined behaviour

  Fix: Added GPU resource requests and limits (explicitly set both requests and limits for nvidia.com/gpu: 1)

5. Single replica deployment

    Issue: replicas: 1

    Impact: No redundancy, no high availability, downtime during rollout

    Fix: Increased to 2 replicas





🟠 HIGH
6. Readiness probe too aggressive

    Issue: initialDelaySeconds: 1

    Impact: Pod may receive traffic before model is fully loaded

    Fix: Increased delay and added proper thresholds

7. Missing startup probe

    Issue: No probe accounting for model load time

    Impact: Container may be restarted before initialization completes

    Fix: Added startupProbe with extended failure threshold

8. CPU-based HPA for GPU workload

    Issue: Autoscaling based solely on CPU utilisation

    Impact: Does not reflect true bottlenecks for inference workloads

    Fix: Retained for simplicity but tuned threshold and documented limitation

    Note: Benchmark data shows GPU compute (~5–6ms) is a small portion of total latency (~30ms), indicating CPU is not the primary scaling signal.
    In production, scaling should use custom metrics such as:

        GPU utilisation (e.g. via NVIDIA DCGM exporter)

        Request latency (p95/p99)

        Queue depth / in-flight requests

9. Use of latest image tag

    Issue: Non-deterministic deployments

    Impact: Breaks reproducibility and rollback safety

    Fix: Replaced with versioned image tag





🟡 MEDIUM

10. No graceful shutdown handling

    Issue: No lifecycle hooks

    Impact: In-flight requests may be terminated abruptly

    Fix: Added preStop hook

    Assumes application correctly handles SIGTERM by stopping new requests and draining existing ones.

11. Missing liveness probe

    Issue: No mechanism to detect and restart unhealthy containers

    Impact: Failed containers may continue serving traffic

    Fix: Added liveness probe

12. Missing GPU node scheduling constraints

    Issue: No node selection for GPU-capable nodes

    Impact: Inefficient or incorrect scheduling

    Fix: Added nodeSelector for GPU nodes

13. CPU limits defined (removed)

    Issue: CPU is a compressible resource but was limited

    Impact: Risk of throttling and increased latency

    Fix: Removed CPU limits

14. Missing security context

    Issue: Default container privileges

    Impact: Increased attack surface

    Fix: Added:

        runAsNonRoot

        allowPrivilegeEscalation: false

        readOnlyRootFilesystem: true




⚪ Not Fixed (with reasoning)

1. HPA uses CPU instead of custom metrics

    Reason: Proper implementation requires external metrics pipeline (Prometheus + custom metrics API)

    Improvement: Use GPU utilisation (DCGM), latency, or queue-based scaling

2. No PodDisruptionBudget (PDB)

    Reason: Out of scope for minimal manifest

    Improvement: Add PDB to ensure availability during node drains

3. No observability configuration (metrics/logging/tracing)

    Reason: Requires broader platform setup

    Improvement: Add Prometheus scraping annotations and /metrics endpoint

4. No namespace specified

    Reason: Often handled by deployment tooling (Helm, ArgoCD)

    Improvement: Use dedicated namespace per environment

5. GPU device-level pinning not configured

    Reason: Kubernetes device plugin handles allocation

    Improvement: Topology-aware scheduling can be added if required




