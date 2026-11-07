FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the compose healthcheck; torch/psycopg wheels need no build deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first so it is cached independently of app changes,
# and so the image never pulls the multi-GB CUDA build.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"

# Install the project and its remaining dependencies.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
RUN pip install .

# Run as a non-root user. The /data and /models dirs are created and owned here;
# an empty named volume mounted at /data inherits this ownership on first use.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/uploads /data/masks /models \
    && chown -R appuser:appuser /data /models /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
