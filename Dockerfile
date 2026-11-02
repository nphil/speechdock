FROM node:22-bookworm-slim AS frontend
WORKDIR /build/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.9.30 AS uv
FROM python:3.12-slim-bookworm AS crisp-runtime
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:772160483fe5fb43024237fac43563f270bed7877d72b3333282024d070bced4 https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.48/crispasr-linux-x86_64-cuda.tar.gz /tmp/crispasr.tar.gz
COPY runtime/ /build/
RUN cc -shared -fPIC -O2 -Wall -Wextra -Werror -o /tmp/cuda-required.so /build/cuda_required.c -ldl \
    && cc -O2 -Wall -Wextra -Werror -o /tmp/cuda-launch /build/cuda_launch.c -ldl \
    && python /build/prepare.py /tmp/crispasr.tar.gz /tmp/cuda-required.so /tmp/cuda-launch /opt/speechdock/runtime/crispasr-0.8.48-cuda12-1

FROM python:3.12-slim-bookworm AS runtime
ARG VERSION=0.2.1
ARG REVISION=unknown
ARG CREATED=unknown
LABEL org.opencontainers.image.title="SpeechDock" \
      org.opencontainers.image.description="Local speech and subtitle workspace" \
      org.opencontainers.image.source="https://github.com/nphil/speechdock" \
      org.opencontainers.image.url="https://github.com/nphil/speechdock" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${CREATED}"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SPDOCK_DATA_DIR=/data SPDOCK_LLAMA_CONFIG=/llama-config/config.yaml \
    SPDOCK_MODELS_DIR=/models-owned SPDOCK_ENGINE_MODELS_DIR=/models/whisper/speechdock \
    SPDOCK_LLAMA_VALIDATOR=/validator/llama-swap SPDOCK_FRONTEND_DIR=/app/ui/dist \
    SPDOCK_ENGINE_URL=https://llama.nateshome.net \
    SPDOCK_ALIGNMENT_MODEL_DIR=/opt/speechdock/models/alignment \
    PATH=/app/.venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /data /models-owned /llama-config /etc/llama-swap \
    && ln -s /llama-config /etc/llama-swap/config \
    && chown 1000:1000 /data /models-owned /llama-config
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
COPY speechdock/ ./speechdock/
RUN uv sync --locked --no-dev --no-editable && rm /usr/local/bin/uv
COPY runtime/prepare_alignment.py runtime/alignment-LICENSE.txt /tmp/alignment-build/
RUN python /tmp/alignment-build/prepare_alignment.py "$SPDOCK_ALIGNMENT_MODEL_DIR" \
    && rm -rf /tmp/alignment-build
COPY --from=frontend /build/ui/dist ./ui/dist
COPY --from=crisp-runtime /opt/speechdock/runtime /opt/speechdock/runtime
ADD --checksum=sha256:2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f https://raw.githubusercontent.com/snakers4/silero-vad/v5.1/src/silero_vad/data/silero_vad.onnx /opt/speechdock/models/silero-vad-v5.1.onnx
RUN chmod 0555 /opt/speechdock /opt/speechdock/models \
    && chmod 0555 /opt/speechdock/models/alignment \
    && chmod 0444 /opt/speechdock/models/silero-vad-v5.1.onnx /opt/speechdock/models/alignment/*
USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]
CMD ["uvicorn", "speechdock.app:app", "--host", "0.0.0.0", "--port", "8000"]
