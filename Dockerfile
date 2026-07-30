# ==============================================================
# tt-embedding-variation
# Base: PyTorch 2.4.1 + CUDA 12.4 (Ubuntu 22.04)
# ==============================================================
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

LABEL maintainer="tt-embedding-variation"
LABEL description="Two-Tower Embedding Diversity Experiment Environment"

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        build-essential \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /workspace

# ---- Python dependencies ----
# Copy requirements first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Project source ----
COPY . .

# ---- HuggingFace cache dir (mount from host) ----
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# ---- FAISS GPU config ----
ENV OMP_NUM_THREADS=8

# ---- Default command: bash ----
CMD ["/bin/bash"]
