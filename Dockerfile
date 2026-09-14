FROM python:3.12-slim

ARG VCS_REF=unknown
ARG MODEL_VERSION=unknown
ARG MODEL_FILE_SHA256=unknown

LABEL org.opencontainers.image.source="https://github.com/jacobmackey01/card-fraud-threshold-modelling" \
    org.opencontainers.image.revision="${VCS_REF}" \
    io.card-fraud.model.version="${MODEL_VERSION}" \
    io.card-fraud.model.file.sha256="${MODEL_FILE_SHA256}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY service ./service
COPY artifacts ./artifacts
COPY examples/synthetic_demo_v1.json ./examples/synthetic_demo_v1.json

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/ready', timeout=2)"

CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT}"]
