# RUDRA HuggingFace Space — Dockerfile
# Uses llama-cpp-python on CPU, no GPU needed.
# Model GGUF must be mounted or downloaded at runtime (set RUDRA_MODEL_PATH).

FROM python:3.11-slim

WORKDIR /app

# Install system deps for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Environment defaults
ENV RUDRA_MODEL_PATH=rudra-q4_k_m.gguf
ENV RUDRA_N_THREADS=4
ENV RUDRA_N_CTX=4096

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

# Run
CMD ["python", "app.py"]