# Signapse inference server
FROM pytorch/pytorch:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    wget \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy everything in
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask \
    numpy \
    pillow \
    torchvision \
    pytest \
    black \
    isort \
    mypy \
    ipython \
    jupyter \
    notebook \
    requests \
    boto3

# Download model weights at build time so they are baked into the image
RUN python -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.DEFAULT)"

EXPOSE 8080

CMD ["python", "app.py"]
