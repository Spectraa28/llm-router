FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache \
    HF_HOME=/app/model_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/

WORKDIR /app/src

RUN mkdir -p cache_store /app/model_cache

# Pre-download model INTO the image at a fixed path
RUN python -c "
from sentence_transformers import SentenceTransformer
import os
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/model_cache')
print('Model cached.')
"

# Pre-train classifier and save the artifact
RUN python -c "from classifier import _train_model; _train_model()"

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]