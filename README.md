# Data Extraction API

## Description
A production-ready FastAPI REST API that ingests Base64-encoded documents (PDF, DOCX, images) and safely returns structured JSON containing AI-generated summaries, named entities, and sentiment analysis. It dynamically falls back to OCR for scanned documents to ensure highly accurate layout preservation and unified extraction.

## Tech Stack
- **Language/Framework:** Python 3.11, FastAPI
- **Key libraries:** PyMuPDF, python-docx, Tesseract (pytesseract), Pydantic
- **LLM/AI models used:** Groq API (Llama 3.3 70B) for Summary, Entity Extraction, and Sentiment Analysis.

## Setup Instructions
1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/documentExtraction.git
   cd documentExtraction
   ```
2. Install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux (use venv\Scripts\activate on Windows)
   pip install -r requirements.txt
   ```
3. Set environment variables
   ```bash
   cp .env.example .env
   # Edit .env and enter your API_KEY and GROQ_API_KEY
   ```
4. Run the application
   **Local Execution:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   **Docker Execution (Recommended for OS-level Tesseract integration):**
   ```bash
   docker-compose up --build
   ```

## Approach
Explain your Data Extraction strategy:
- **How we extract text:** We natively parse PDF binaries via PyMuPDF and DOCX XML via `python-docx` for optimal layout retention and lightning speed. If an image or scanned PDF is detected, the extraction pipeline dynamically redirects the byte streams directly through Tesseract OCR to interpret the visual layout.
- **How we extract summary, entities and analyze sentiment:** Instead of running heavy local NLP dependencies that block asynchronous routes, we serialize the extracted text directly to Groq's high-speed cloud inference engine using structured, constrained JSON-mode prompting. This grants us absolute control over Entity recognition (strictly filtering out roles/headings), provides world-class Abstractive Summarization, and effortlessly classifies Sentiment logic, guaranteeing compliance with scale limitations and flawless, untempered responses.

---
# guvi

# AI Used :

Claude Sonnet 4.5
Claude Opus 4.5
Gemini 3.1 Pro