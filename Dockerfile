# HuggingFace Jobs / RunPod / Vast.ai compatible GPU image
# Base: nvidia/cuda on Ubuntu 24.04 (ships Python 3.12 natively)
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    libexpat1 libgeos-dev libproj-dev libgdal-dev \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first for layer caching
COPY pyproject.toml uv.lock* ./

# uv config (https://docs.astral.sh/uv/guides/integration/docker/)
ENV UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy

RUN uv sync --extra gpu --no-dev --no-cache

# Application code
COPY pipeline/ ./pipeline/
COPY main.py ./

# Pre-install DuckDB extensions (cached in image, no download at runtime)
RUN uv run python -c "import duckdb; con = duckdb.connect(); con.install_extension('httpfs'); con.install_extension('h3', repository='community'); print('DuckDB extensions installed')"

# Smoke-test imports at build time
RUN uv run python -c "from pipeline import config, gpu, h3_grid; print('imports OK')"

# Default: idle health endpoint on port 7860 so the Space stays "Running"
# without executing the pipeline.  HF Jobs override CMD with main.py.
CMD ["python3", "-c", "from http.server import HTTPServer, BaseHTTPRequestHandler\nHTTPServer(('', 7860), BaseHTTPRequestHandler).serve_forever()"]
