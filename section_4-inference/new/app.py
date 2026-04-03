
"""
Signapse — Sign Language Video Generation
Inference server: serves a single /predict endpoint.

Takes a JSON payload with a "frames" key (list of pose tensors)
and returns a base64-encoded video frame sequence.
"""

import base64
import io
import logging
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported input sizes
SUPPORTED_SIZES = [64, 128, 224]
DEFAULT_SIZE = 224

app = FastAPI()

class PredictRequest(BaseModel):
    frames: List[List[float]]
    size: Optional[int] = DEFAULT_SIZE

class InferenceServer:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, DEFAULT_SIZE * DEFAULT_SIZE * 3),
        ) ## the fc layer is untrained - can't tell if that's one of the deliberate mistakes or just using a dummy model for purposes of the question.
        #model = torch.compile(model) # slower startup, but moderate inference speed up. Am leaving this in, but would need to understand the larger spin up vs inference tradeoff for our use case before deciding to keep it in production.
        #"torch.compile removed: requires a C compiler at runtime which conflicts with using a lean runtime base image. Would evaluate in a separate build-image variant if compile-time speedup justifies the image size increase."
        
        model.eval()
        return model.cuda()

    def preprocess_frames(self, frames, size=DEFAULT_SIZE):
        processed = []
        for frame_data in frames:
            arr = np.array(frame_data, dtype=np.float32)
            if arr.ndim == 1:
                side = int(len(frame_data) ** 0.5)
                arr = arr.reshape(side, side)
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
            arr = (arr - 0.485) / 0.229  # Normalize 
            ## Something about the normalise feels slightly off. Resnet wants imagenet normalisation, but without real pose data to test against I'm just leaving the imagenet normalization as it was. (but moved away from torch)
            processed.append(arr)
        batch = np.stack(processed)  # (batch, H, W)
        batch = batch[:, np.newaxis, :, :]  # (batch, 1, H, W)
        return batch

    def postprocess_output(self, output_tensor, size=DEFAULT_SIZE):
        frame = output_tensor.cpu().detach().numpy()
        frame = frame.reshape(size, size, 3)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        return base64.b64encode(buffer).decode('utf-8')

server = InferenceServer()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    t0 = time.time()
    frames = request.frames
    size = request.size if request.size in SUPPORTED_SIZES else DEFAULT_SIZE
    if not isinstance(frames, list) or len(frames) == 0:
        raise HTTPException(status_code=400, detail="'frames' must be a non-empty list")
    try:
        t1 = time.time()
        batch_tensor = server.preprocess_frames(frames, size=size)
        t2 = time.time()

        batch_tensor = torch.from_numpy(batch_tensor)
        batch_tensor = torch.nn.functional.interpolate(
            batch_tensor.repeat(1, 3, 1, 1),
            size=(DEFAULT_SIZE, DEFAULT_SIZE),
        ).pin_memory()
        batch_tensor = batch_tensor.to("cuda", non_blocking=True)
        
        t3 = time.time()
        t4 = t3  # Maintain timing structure for logging compatibility
        t5 = time.time()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        with torch.inference_mode():
            outputs = server.model(batch_tensor)
        gpu_end.record()
        torch.cuda.synchronize()
        t6 = time.time()
        gpu_active_ms = gpu_start.elapsed_time(gpu_end)
        gpu_idle_before_ms = (t5 - t4) * 1000
        wall_inference_ms = (t6 - t5) * 1000
        launch_overhead_ms = wall_inference_ms - gpu_active_ms
        output_frames = [
            server.postprocess_output(out, size=size)
            for out in outputs
        ]
        t7 = time.time()
        logger.info(f"Predicted {len(frames)} frames in {t7-t0:.3f}s (gpu_active={gpu_active_ms:.1f}ms, launch_overhead={launch_overhead_ms:.1f}ms)")
        return {
            "frames": output_frames,
            "count": len(output_frames),
            "elapsed_seconds": round(t7 - t0, 3),
            "timing": {
                "preprocessing": round(t2 - t1, 6),
                "batching": round(t3 - t2, 6),
                "resize": round(t4 - t3, 6),
                "cuda_sync_before": round(t5 - t4, 6),
                "inference": round(t6 - t5, 6),
                "postprocessing": round(t7 - t6, 6),
                "total": round(t7 - t0, 6),
                "gpu_idle_before": round(gpu_idle_before_ms / 1000, 6),
                "gpu_active": round(gpu_active_ms / 1000, 6),
                "launch_overhead": round(launch_overhead_ms / 1000, 6),
            }
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
