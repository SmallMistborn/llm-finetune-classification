FROM python:3.12-slim-bookworm

# cu124 — CUDA 12.4 wheels (совместимо с RTX 4090 / драйвер 550+)
# CPU-сборка: docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.12.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SFT_OUTPUT=/app/sft_output \
    SFT_OUTPUT_PEFT=/app/sft_output_peft \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1) torch с CUDA (или CPU, если TORCH_INDEX_URL=.../cpu)
# 2) остальные зависимости без повторной установки torch с PyPI
RUN pip install --upgrade pip \
    && pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && grep -vE '^\s*torch==' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

COPY src/ ./src/

RUN mkdir -p /app/sft_output /app/sft_output_peft /app/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
