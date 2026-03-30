"""
Signapse — Sign Language Video Generation
Inference server: serves a single /predict endpoint.

Takes a JSON payload with a "frames" key (list of pose tensors)
and returns a base64-encoded video frame sequence.
"""

import base64
import io
import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported input sizes
SUPPORTED_SIZES = [64, 128, 224]
DEFAULT_SIZE = 224


def get_model():
    """Load the frame generation model."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace final layer for our frame generation task
    model.fc = nn.Sequential(
        nn.Linear(2048, 4096),
        nn.ReLU(),
        nn.Linear(4096, DEFAULT_SIZE * DEFAULT_SIZE * 3),
    )

    model.eval()
    return model


def preprocess_frame(frame_data, size=DEFAULT_SIZE):
    """Convert raw pose tensor data to model input."""
    tensor = torch.tensor(frame_data, dtype=torch.float32)

    if tensor.dim() == 1:
        side = int(len(frame_data) ** 0.5)
        tensor = tensor.reshape(1, 1, side, side)

    tensor = tensor.to("cuda")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    # Move to CPU for transform, back to GPU for inference
    tensor = tensor.cpu()
    tensor = transform(tensor)
    tensor = tensor.cuda()

    return tensor


def postprocess_output(output_tensor, size=DEFAULT_SIZE):
    """Convert raw model output to a serialisable frame."""
    # Pull back to CPU for numpy conversion
    frame = output_tensor.cpu().detach().numpy()
    frame = frame.reshape(size, size, 3)

    # Normalise to 0-255
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
    frame = (frame * 255).astype(np.uint8)

    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()

    data = request.get_json()
    if not data or "frames" not in data:
        return jsonify({"error": "Missing 'frames' in request body"}), 400

    frames = data["frames"]
    if not isinstance(frames, list) or len(frames) == 0:
        return jsonify({"error": "'frames' must be a non-empty list"}), 400

    size = data.get("size", DEFAULT_SIZE)
    if size not in SUPPORTED_SIZES:
        return jsonify({"error": f"'size' must be one of {SUPPORTED_SIZES}"}), 400

    try:
        # Load model fresh for each request
        model = get_model()
        model = model.cuda()

        output_frames = []

        for frame_data in frames:
            input_tensor = preprocess_frame(frame_data, size=size)

            # Pad to batch dimension
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

            # Resize to match resnet expected input
            input_tensor = torch.nn.functional.interpolate(
                input_tensor.expand(-1, 3, -1, -1),
                size=(DEFAULT_SIZE, DEFAULT_SIZE),
            )

            # Sync before inference to ensure input is ready
            torch.cuda.synchronize()

            output = model(input_tensor)

            # Sync after inference to ensure output is complete
            torch.cuda.synchronize()

            encoded = postprocess_output(output, size=size)
            output_frames.append(encoded)

        elapsed = time.time() - start
        logger.info(f"Predicted {len(frames)} frames in {elapsed:.3f}s")

        return jsonify({
            "frames": output_frames,
            "count": len(output_frames),
            "elapsed_seconds": round(elapsed, 3),
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting inference server...")
    app.run(host="0.0.0.0", port=8080, debug=True)
