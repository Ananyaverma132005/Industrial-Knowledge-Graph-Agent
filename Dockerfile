# Container image for hosting (Hugging Face Spaces, Render, Railway, any Docker host).
# Builds the knowledge graph, SOP vectors and ML model at build time, so the container
# starts ready to serve. Runs in fallback mode (in-memory graph, extractive SOP answers)
# unless NEO4J_* / OLLAMA_* environment variables point at real services.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm

COPY . .

# Build-time data pipeline: sample manuals -> ML model -> graph -> SOP vectors.
RUN python make_sample_manuals.py \
 && python train_model.py \
 && python ingest.py \
 && python ingest_pdf.py

# The embedding model is now cached in the image; don't re-check the Hub at startup.
ENV HF_HUB_OFFLINE=1

# Hugging Face Spaces runs containers as user 1000.
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
