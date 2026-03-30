# Signapse — ML Systems Engineer Technical Assessment

> **Confidential** · Please do not share or publish this assessment.

Thank you for progressing to the technical stage. This test reflects the kinds of problems you would actually work on at Signapse reducing inference latency, scaling GPU infrastructure, and building reliable production ML systems.

**Estimated time:** 3–4 hours total. There is no hard deadline — quality of thinking matters more than speed.

**Submission:** Return your written answers as a single document (PDF, Word, or Markdown), and include any code files alongside it in this repo.

---

## What we are looking for

- **Production instincts** — what would you actually do in a real system?
- **Structured reasoning** — show your working, not just your conclusions
- **Honest self-assessment** — tell us where your knowledge runs out; there are no trick questions
- **Clear communication** — write as if explaining to a peer engineer

## Section 1 — ML Inference & Optimisation

*Estimated time: ~60 minutes*

A senior engineer suspects the 18-second latency is not caused by model weight size, but by inefficiencies in the inference code — specifically unnecessary CPU/GPU transfers and unoptimised data movement within the PyTorch pipeline.

### 1.1 Profiling and fixing the pipeline

You have SSH access to the inference server and the PyTorch source code. Describe how you would approach this end-to-end:

- Which tools you would use to profile the pipeline and why
- How you would distinguish between CPU-bound, GPU-bound, and I/O-bound hotspots
- The optimisation techniques you would apply, in priority order — for each one, briefly note what it addresses, what the trade-off is, and how you would confirm it did not degrade output quality

### 1.2 ONNX / TensorRT conversion

The team wants to trial exporting the video generation model to ONNX and running it via TensorRT. Walk through how you would do this safely:

- How you would handle dynamic input shapes (variable sentence lengths)
- How you would validate the converted model produces equivalent outputs
- How you would decide whether TensorRT actually gives a meaningful speedup for this specific workload

> **Note:** If you have not used TensorRT directly, explain how you would approach learning it and what you would look for first.

### 1.3 Incident

At 11pm on a Tuesday, inference latency spikes from 7s to 45s on the live streaming pipeline. Grafana shows GPU utilisation has dropped from 85% to 12%.

Walk through your diagnosis: what you check first, how you distinguish between a model issue, an infrastructure issue, and a traffic issue, and what you do to restore service before you have a root cause.

---

## Section 2 — Kubernetes Practical

*Estimated time: ~45 minutes*

The files for this section are in the [`k8s/`](./k8s/) directory:

```
k8s/
└── inference-deployment.yaml
```

This manifest was written by an engineer who was more familiar with local model serving than production Kubernetes. It will deploy — but it has a collection of correctness problems, security issues, and operational bad practices that would cause real problems in production, particularly for a GPU inference workload.

**Your task:** review the manifest, identify every problem you can find, fix them, and document what you changed and why in a short written changelog — ordered by severity.

### What to submit

- Your corrected `inference-deployment.yaml`
- A written changelog: each issue, why it matters, and what you did about it
- Any issues you spotted but chose not to fix — note those too with your reasoning

### Hints — what to look for

Problems exist across these categories:

- **GPU scheduling** — the manifest does not correctly request or constrain GPU resources
- **Reliability** — configuration choices that will cause downtime or drop in-flight requests
- **Security** — credentials that should not be where they are
- **Autoscaling** — the HPA is configured in a way that will not work well for this workload
- **Operational** — missing things that will make this hard to operate safely in production

There are at least **8 distinct issues**. Some are one-liners, some require a small addition.

> **Note:** You do not need a running cluster to complete this. Reading the manifest carefully and reasoning through the implications is sufficient.

### Written question

Once you have fixed the manifest: the current deployment allocates one full GPU per pod despite the model weights being under 1GB. Describe your approach to packing multiple model instances onto a single GPU — covering how you would choose between MIG and time-slicing for this workload, what changes to the manifest that requires, and how you would test it safely before rolling to production.

---

## Section 3 — Code Task

*Estimated time: ~60 minutes*

Write a Python script that:

- Loads a small PyTorch model (`torchvision.models.resnet18` as a stand-in for our video model)
- Runs a configurable number of warm-up passes followed by a configurable number of timed passes
- Records per-pass latency and reports **p50, p95, p99, mean, and standard deviation**
- Reports whether the bottleneck appears to be GPU-bound or CPU-bound based on utilisation (use any method — `psutil`, `pynvml`, `nvidia-smi` subprocess, etc.)
- Produces a clean summary table to stdout

### Bonus (optional)

- Export the model to ONNX and run the same benchmark, printing a side-by-side comparison
- Flag passes more than 2 standard deviations above the mean as outliers

> **Note:** You do not need to run this against real hardware — correctness of logic is what we are assessing. Submit it as `profiler.py` in this repo.

---

## Section 4 — Code Review: Broken Inference Server

*Estimated time: ~60 minutes*

The files for this section are in the [`inference/`](./inference/) directory:

```
inference/
├── app.py          # Flask inference server
└── Dockerfile      # Container definition
```

The code was written by an engineer more familiar with research environments than production ML systems. It works — in the sense that it will serve predictions — but it contains correctness issues, performance problems, and operational bad practices.

**Your task:** review both files, fix every problem you can find, and document what you changed and why.

### What to submit

- Your corrected `app.py` and `Dockerfile`
- A written changelog: each issue, why it matters, what you did — ordered by severity

### Hints — what to look for

Problems exist across these categories in both files:

- **Correctness** — code that will produce wrong results or fail silently
- **Performance** — unnecessary work on every request
- **GPU hygiene** — misuse of device transfers and synchronisation
- **Container best practices** — image size, caching, security, reproducibility
- **Production readiness** — things that would cause problems under real load

There are at least **8 distinct issues** across the two files.

> **Note:** You do not need a GPU to complete this. Reasoning from the source is a valid approach.

---

## Final question

Which part of this test felt most outside your current experience? What is your honest plan for closing that gap in the first 90 days at Signapse?

*(There is no wrong answer — we value self-awareness over false confidence.)*

---

## Submission checklist

- [ ] Written answers for Sections 1 and 2 (Markdown, PDF, or Word)
- [ ] Corrected `inference-deployment.yaml` for Section 2
- [ ] `profiler.py` for Section 3
- [ ] Corrected `app.py` and `Dockerfile` for Section 4
- [ ] Written changelogs for Sections 2 and 4
- [ ] Final question answered
- [ ] Sent to [careers@signapse.ai](mailto:careers@signapse.ai) with subject `Technical Test — [Your Name]` or your hiring manager.