# Signapse — ML Systems Engineer Technical Assessment

> **Confidential** · Please do not share or publish this assessment.

Thank you for progressing to the technical stage. This test is designed to reflect the kinds of problems you would actually work on at Signapse reducing inference latency, scaling GPU infrastructure, and building reliable production ML systems.

**Estimated time:** 4–5 hours total. There is no hard deadline quality of thinking matters more than speed.

**Submission:** Return your answers as a single document (PDF, Word, or Markdown). Code can be included inline or as a `.py` file in this repo. Send to [careers@signapse.ai](mailto:careers@signapse.ai) with the subject line: `Technical Test — [Your Name]`.

---

## What we are looking for

- **Structured reasoning** — show your working, not just your conclusions
- **Production instincts** — what would you actually do in a real system?
- **Honest self-assessment** — tell us where your knowledge runs out; there are no trick questions
- **Clear communication** — write as if explaining to a peer engineer

---

## Section 1 — ML Inference Profiling & Optimisation

*Estimated time: ~60 minutes*

A senior engineer suspects the 18-second latency is not caused by model weight size, but by inefficiencies in the inference code specifically unnecessary CPU/GPU transfers and unoptimised data movement within the PyTorch pipeline.

### 1.1 Profiling strategy

You have SSH access to the inference server and the PyTorch source code. Describe step-by-step how you would profile the pipeline to identify the top bottlenecks. Be specific about:

- Which tools or libraries you would use and why
- What metrics you would look at first
- How you would distinguish between CPU-bound, GPU-bound, and I/O-bound hotspots
- What a suspicious CPU/GPU transfer looks like in a profile trace

### 1.2 Optimisation plan

Once you have identified the hotspots, describe the optimisation techniques you would consider applying in priority order. For each technique, briefly explain:

- What problem it addresses
- What the risk or trade-off is
- How you would verify it did not degrade output quality

### 1.3 ONNX / TensorRT conversion

The team wants to trial exporting the video generation model to ONNX and running it via TensorRT. Walk through the steps you would take to do this safely, including:

- How you would handle dynamic input shapes (variable sentence lengths)
- How you would validate the converted model produces equivalent outputs
- What failure modes you would watch out for during conversion
- How you would decide whether TensorRT actually gives a meaningful speedup for this workload

> **Note:** If you have not used TensorRT directly, explain the approach you would take to learn it quickly and what you would look for in the documentation.

---

## Section 2 — Kubernetes & GPU Scheduling

*Estimated time: ~60 minutes*

Signapse's inference workloads run on Kubernetes clusters (cloud and on-prem). Currently, each model requires a full GPU per pod despite the model weights being under 1GB — which makes it expensive to scale to 100 concurrent live streams. The team is investigating fractional GPU sharing and multi-model packing strategies.

### 2.1 Kubernetes fundamentals

Explain the following concepts in your own words, as you would to a colleague who is a strong ML engineer but new to Kubernetes:

- What a Pod, Deployment, and Service are, and how they relate to each other
- How Kubernetes resource requests and limits work for GPU workloads
- What happens when a node runs out of GPU memory mid-inference

> **Note:** If your Kubernetes experience is limited, be specific about what you do and do not know — and explain how you would get up to speed.

### 2.2 GPU sharing problem

Given that the current model fits in under 1GB of GPU memory but is allocated an entire GPU per pod, design an approach to pack multiple model instances onto a single GPU. Your answer should cover:

- How you would evaluate whether MIG (Multi-Instance GPU) or time-slicing is more appropriate here
- What changes would be needed in the Kubernetes manifests
- What new failure modes this introduces and how you would monitor for them
- How you would test this safely before rolling it out to production

### 2.3 Autoscaling for real-time SLAs

Describe how you would configure autoscaling for the inference deployment to handle bursts to 100 concurrent streams while respecting a p95 latency SLA. Consider:

- What metric you would use to trigger scaling (CPU, GPU utilisation, queue depth, custom metric?)
- How you would avoid cold-start latency spikes when new pods spin up
- What happens if the cluster hits its GPU node limit during a burst

---

## Section 3 — Benchmarking & Observability

*Estimated time: ~45 minutes*

The team currently uses Prometheus and Grafana for infrastructure metrics. There is no systematic benchmarking framework for model inference performance, which makes it difficult to know whether optimisation changes are actually working.

### 3.1 Benchmarking framework design

Design a benchmarking framework for the sign language video generation pipeline. Your design should specify:

- What inputs you would use (fixed test set, synthetic, sampled from production?)
- Which latency metrics to track (p50, p95, p99, time-to-first-frame, end-to-end?)
- How you would separate model inference time from pre/post-processing and I/O
- How results would be stored and compared across model versions
- How you would prevent benchmark results from becoming stale or misleading over time

### 3.2 Production alerting

Define the alerting rules you would set up in Prometheus/Grafana for this inference service. For each alert, specify:

- The metric and threshold
- The severity and who gets paged
- How you would avoid alert fatigue and false positives

### 3.3 Incident diagnosis

At 11pm on a Tuesday, inference latency spikes from 7s to 45s on the live streaming pipeline. Grafana shows GPU utilisation has dropped from 85% to 12%. Walk through your diagnosis process:

- What you check first and why
- How you distinguish between a model issue, an infrastructure issue, and a traffic pattern issue
- What actions you would take to restore service without waiting for a root cause
- What you would document afterwards

---

## Section 4 — Code Task

*Estimated time: ~60 minutes*

Write a Python script that:

- Loads a small PyTorch model (use `torchvision.models.resnet18` as a stand-in for our video model)
- Runs a configurable number of warm-up passes followed by a configurable number of timed passes
- Records per-pass latency and reports **p50, p95, p99, mean, and standard deviation**
- Reports whether the bottleneck appears to be GPU-bound or CPU-bound based on utilisation (use any method you like — `psutil`, `pynvml`, `nvidia-smi` subprocess, etc.)
- Produces a simple summary table printed to stdout

### Bonus (optional)

If you have time:

- Export the model to ONNX and run the same benchmark, printing a side-by-side comparison
- Flag any passes where latency is more than 2 standard deviations above the mean as potential outliers

> **Note:** You do not need to run this against real hardware — correctness of logic is what we are assessing. Include the script inline in your answers document or as a `.py` file in this repo.

A starter scaffold is provided in [`scaffold.py`](./scaffold.py) if you find it useful — you are not required to use it.

---

## Section 5 — Code Review: Broken Inference Server

*Estimated time: ~60 minutes*

This section gives you real (deliberately broken) code to fix. The files are in the [`inference/`](./inference/) directory of this repo:

```
inference/
├── app.py          # Flask inference server
└── Dockerfile      # Container definition
```

The code was written by an engineer who was more familiar with research environments than production ML systems. It works — in the sense that it will serve predictions — but it has a collection of correctness issues, performance problems, and operational bad practices baked in.

**Your task:** review both files, identify every problem you can find, fix them, and document what you changed and why.

### What to submit

- Your corrected `app.py` and `Dockerfile`
- A written changelog explaining each issue you found, why it matters, and what you did about it — ordered by severity if possible
- If there are issues you spotted but chose not to fix (e.g. out of scope, would require architectural changes), note those too

### Hints — what to look for

We are not going to tell you exactly what is wrong. But broadly, issues exist across these categories in both files:

- **Correctness** — code that will produce wrong results or fail silently under real conditions
- **Performance** — unnecessary work being done on every request
- **GPU hygiene** — misuse of device transfers and synchronisation
- **Container best practices** — image size, caching, security, and reproducibility
- **Production readiness** — things that would cause problems in a live system

There are at least **8 distinct issues** across the two files. Some are obvious, some are subtle.

> **Note:** You do not need a GPU to complete this section. If you cannot run the code locally, reason through the issues from reading the source — that is a valid and expected approach.

---

## Final question

Looking across all four sections: which part of this test felt most outside your current experience? What is your honest plan for closing that gap if you joined Signapse?

*(There is no wrong answer — we value self-awareness over false confidence.)*

---

## Submission checklist

- [ ] Answers to Sections 1–3 (Markdown, PDF, or Word)
- [ ] Code for Section 4 (inline or as `solution.py`)
- [ ] Corrected `app.py` and `Dockerfile` for Section 5
- [ ] Written changelog for Section 5 fixes
- [ ] Final question answered
- [ ] Sent to [careers@signapse.ai](mailto:careers@signapse.ai) with subject `Technical Test — [Your Name]`

We will review your submission and be in touch within 5 working days. If you have any questions in the meantime, please reach out to [careers@signapse.ai](mailto:careers@signapse.ai).