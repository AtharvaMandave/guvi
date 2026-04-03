# AI-Powered Document Analysis & Extraction API

A production-ready **FastAPI** REST API that ingests Base64-encoded documents (PDF, DOCX, images) and returns structured JSON with:

- 📝 **Summary** — AI-generated concise summary via Groq (Llama 3.3 70B)
- 🏷️ **Named Entities** — Person names, dates, organizations, monetary amounts via spaCy
- 💬 **Sentiment** — Positive / Negative / Neutral classification via Groq

## Architecture

```
Client (JSON + Base64) → FastAPI → Auth → Text Extraction → NLP Pipeline → JSON Response
                                          ├── PDF  (PyMuPDF)        ├── Summarization (Groq)
                                          ├── DOCX (python-docx)    ├── NER (spaCy)
                                          └── Image (Tesseract)     └── Sentiment (Groq)
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
- [Groq API key](https://console.groq.com) (free, no credit card required)

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/documentExtraction.git
cd documentExtraction

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Configure environment
copy .env.example .env
# Edit .env → set your API_KEY and GROQ_API_KEY
```

### 3. Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 4. Run with Docker

```bash
docker-compose up --build
```

The API will be available at **http://localhost:8000**.

---

## API Reference

### Health Check

```
GET /
```

Response: `{"status": "alive", "service": "Document Analysis API", "version": "1.0.0"}`

### Analyze Document

```
POST /document/analyze
```

**Headers:**
| Header | Required | Description |
|--------|----------|-------------|
| `x-api-key` | Yes | Your API key (set in `.env`) |
| `Content-Type` | Yes | `application/json` |

**Request Body:**
```json
{
  "fileName": "report.pdf",
  "fileType": "pdf",
  "fileBase64": "<base64-encoded-file-content>"
}
```

**Supported File Types:** `pdf`, `docx`, `png`, `jpg`, `jpeg`, `tiff`, `bmp`

**Success Response (200):**
```json
{
  "status": "success",
  "fileName": "report.pdf",
  "summary": "The report discusses quarterly earnings showing a 15% revenue increase...",
  "entities": {
    "names": ["John Smith", "Jane Doe"],
    "dates": ["January 2025", "Q4 2024"],
    "organizations": ["Acme Corp", "TechStart Inc"],
    "amounts": ["$1.5 million", "$500,000"]
  },
  "sentiment": "Positive"
}
```

### Example cURL Request

```bash
# Encode a file to Base64
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("report.pdf"))

# Send request (PowerShell)
Invoke-RestMethod -Uri "http://localhost:8000/document/analyze" `
  -Method POST `
  -Headers @{ "x-api-key" = "your-api-key"; "Content-Type" = "application/json" } `
  -Body (@{
    fileName = "report.pdf"
    fileType = "pdf"
    fileBase64 = $b64
  } | ConvertTo-Json)
```

```bash
# Linux/macOS
BASE64=$(base64 -w 0 report.pdf)
curl -X POST http://localhost:8000/document/analyze \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d "{\"fileName\": \"report.pdf\", \"fileType\": \"pdf\", \"fileBase64\": \"$BASE64\"}"
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_nlp.py -v
```

---

## Deployment (Render.com)

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set **Environment** to **Docker**
5. Add environment variables: `API_KEY`, `GROQ_API_KEY`
6. Deploy — Render will build the Docker image automatically

**Public URL:** `https://your-service.onrender.com/document/analyze`

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | FastAPI | Async REST API |
| PDF Parser | PyMuPDF (fitz) | Native PDF text extraction |
| DOCX Parser | python-docx | Word document parsing |
| OCR | Tesseract + pytesseract | Image/scanned PDF text |
| NER | spaCy (en_core_web_sm) | Named entity recognition |
| Summarization | Groq API (Llama 3.3 70B) | AI summarization |
| Sentiment | Groq API (Llama 3.3 70B) | Sentiment classification |
| Container | Docker | Reproducible deployment |

---

## License

MIT
# guvi
