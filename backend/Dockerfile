# Use NVIDIA CUDA 12.6 base image with Python 3.12
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.12 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Clone and install SAM 3 (official Facebook Research repository)
RUN git clone https://github.com/facebookresearch/sam3.git /tmp/sam3 && \
    cd /tmp/sam3 && \
    pip install -e .

# Copy parent repository requirements first
COPY requirements.txt /app/parent_requirements.txt
COPY requirements.p3d.txt /app/requirements.p3d.txt
COPY requirements.inference.txt /app/requirements.inference.txt

# Install PyTorch 2.7 and Torchvision with CUDA 12.6 support
RUN pip install --no-cache-dir \
    torch==2.7.0+cu126 \
    torchvision==0.20.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126 || \
    pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install parent repository dependencies
RUN pip install --no-cache-dir -r /app/parent_requirements.txt || true

# Copy backend requirements and install
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the entire repository
COPY . /app/

# Install the SAM 3D Objects package
RUN pip install -e .

# Copy frontend build (if exists)
# The frontend will be built separately and copied to frontend/dist
COPY frontend/dist /app/frontend/dist

# Expose port
EXPOSE 8000

# Environment variables for API keys (to be set at runtime)
ENV HF_TOKEN="" \
    KIE_API_KEY="" \
    PORT=8000 \
    HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Run the server
CMD ["python", "backend/server.py"]
