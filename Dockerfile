FROM node:22-bookworm-slim AS frontend
WORKDIR /build/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.9.30 AS uv
FROM python:3.12-slim-bookworm AS runtime
ARG VERSION=0.1.3
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
COPY --from=frontend /build/ui/dist ./ui/dist
ADD --checksum=sha256:2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f https://raw.githubusercontent.com/snakers4/silero-vad/v5.1/src/silero_vad/data/silero_vad.onnx /opt/speechdock/models/silero-vad-v5.1.onnx
RUN chmod 0555 /opt/speechdock /opt/speechdock/models \
    && chmod 0444 /opt/speechdock/models/silero-vad-v5.1.onnx
USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]
CMD ["uvicorn", "speechdock.app:app", "--host", "0.0.0.0", "--port", "8000"]
