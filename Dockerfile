# Document-QA app image. Runs the same code as local: retrieval -> 2-stage abstention
# -> generation. The LLM is served by a separate Ollama container (see docker-compose);
# this image holds the Python package, the embedder/reranker (downloaded to a cache
# volume at first run), Tesseract for OCR, and the sample corpus + golden set under data/.
FROM python:3.11-slim

# Tesseract with Turkish + English (OCR at ingest); ca-certificates for HF/model
# downloads over TLS; curl for the entrypoint's Ollama healthcheck wait.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng \
        ca-certificates curl \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# libgl1 + libglib2.0-0: opencv-python (cv2, used by ingest preprocessing) needs them
# at runtime and the slim base omits them.

WORKDIR /app

# CPU-only torch first: sentence-transformers would otherwise pull the ~2 GB CUDA wheel,
# and the on-prem target has no GPU (reranker is device=cpu pinned anyway).
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Editable install (like local `.venv/bin/pip install -e .`): the `app` package keeps
# living under /app/src, so config.ROOT (= config.py parents[2]) resolves to /app and
# finds the baked data/ next to it. A plain `pip install .` would relocate the package
# into site-packages and ROOT would point there, missing the index.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# The rest of the repo: measurement scripts under eval/, tests, DEVLOG/TESTING, and data/
# (samples + golden set). .dockerignore keeps build noise out.
COPY . .

ENV HF_HOME=/models \
    OLLAMA_HOST=http://ollama:11434 \
    PYTHONUNBUFFERED=1

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Idle by default: compose overrides CMD with the Streamlit web UI; a standalone
# `docker run` stays idle and the CLI runs via `docker compose exec app python -m app.cli ...`.
CMD ["sleep", "infinity"]
