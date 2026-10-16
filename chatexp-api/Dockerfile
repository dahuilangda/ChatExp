FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG INSTALL_TORCH=1

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      python -m pip install torch --index-url ${TORCH_INDEX_URL}; \
    fi

COPY app /app/app
COPY README.md /app/README.md

ENV PYTHONPATH=/app

EXPOSE 8000

# Liveness of the API image (curl is installed above). Worker/beat services
# built from this same Dockerfile override this with a celery healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
