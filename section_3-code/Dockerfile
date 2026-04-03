FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    wget \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    python3.12-venv \
 && rm -rf /var/lib/apt/lists/*


# moved to a venv to deal with some onnx faff.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# onnx is being a pain to install, so we'll put it in a separate file to avoid invalidating the cache for the rest of the deps when we need to change it.

COPY requirements-onnx.txt .  
RUN pip install --no-cache-dir -r requirements-onnx.txt

COPY . .



## Env variables are required to get onnx-gpu working (it works on my machine at least)

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:\
/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:\
/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:\
/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:\
/usr/local/nvidia/lib:/usr/local/nvidia/lib64





RUN python -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.DEFAULT)"

EXPOSE 8080
CMD ["python", "app.py"]




