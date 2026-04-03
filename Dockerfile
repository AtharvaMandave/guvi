FROM python:3.11-slim

# ── System dependencies (Tesseract OCR) ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download spaCy model ────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Application code ────────────────────────────────────────────────────────
COPY . .

EXPOSE 10000

# ── Start server ────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
