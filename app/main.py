"""
FastAPI application — AI-Powered Document Analysis & Extraction API.

Endpoints:
  GET  /              → Health check
  GET  /health        → Detailed health with model status
  POST /document/analyze → Main analysis endpoint (API key required)
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from app.auth import verify_api_key
from app.config import get_settings
from app.extractor import extract_text, SUPPORTED_TYPES
from app.models import DocumentRequest, DocumentResponse, EntitiesResponse, ErrorResponse
from app.nlp import analyze_sentiment, extract_entities, generate_summary, preload_models

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load NLP models on startup so the first request isn't slow."""
    logger.info("🚀 Starting up — loading models …")
    t0 = time.perf_counter()
    preload_models()
    logger.info("✅ Models loaded in %.1f s", time.perf_counter() - t0)
    yield
    logger.info("👋 Shutting down")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document Analysis & Extraction API",
    description=(
        "AI-powered REST API that ingests Base64-encoded documents "
        "(PDF, DOCX, images) and returns structured JSON with summary, "
        "named entities, and sentiment analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for testing / judging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health endpoints ─────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def root():
    """Serve the Web UI for Document Analysis."""
    import os
    # Serve index.html if it exists, otherwise fallback to health check
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "alive", "service": "Document Analysis API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check showing model readiness."""
    settings = get_settings()
    return {
        "status": "healthy",
        "spacy_model": settings.SPACY_MODEL,
        "groq_model": settings.GROQ_MODEL,
        "groq_configured": bool(settings.GROQ_API_KEY),
        "supported_file_types": sorted(SUPPORTED_TYPES),
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
    }


# ── Main analysis endpoint ───────────────────────────────────────────────────


@app.post(
    "/api/document-analyze",
    response_model=DocumentResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Analysis"],
)
@app.post(
    "/document/analyze",
    response_model=DocumentResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Analysis"],
    summary="Analyze a document",
    description=(
        "Accept a Base64-encoded file (PDF, DOCX, or image), "
        "extract text, and return summary, named entities, and sentiment."
    ),
)
async def analyze_document(
    request: DocumentRequest,
    api_key: str = Depends(verify_api_key),
):
    """Process a document through the full extraction → NLP pipeline."""
    t0 = time.perf_counter()
    file_name = request.fileName
    file_type = request.fileType

    # ── Validate file size ───────────────────────────────────────────────
    settings = get_settings()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(request.fileBase64) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB} MB",
        )

    # ── Validate file type ───────────────────────────────────────────────
    ft = file_type.lower().strip().strip(".")
    if ft not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_type}'. Supported: {', '.join(sorted(SUPPORTED_TYPES))}",
        )

    try:
        # Step 1 — Extract text (CPU-bound → run in thread pool)
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, extract_text, request.fileBase64, file_type
        )

        if not text or not text.strip():
            logger.warning("No text extracted from %s", file_name)
            return DocumentResponse(
                status="success",
                fileName=file_name,
                summary="No text could be extracted from the document.",
                entities=EntitiesResponse(),
                sentiment="Neutral",
            )

        logger.info("Extracted %d chars from %s", len(text), file_name)

        # Step 2 — Run NLP pipeline concurrently
        summary_task = loop.run_in_executor(None, generate_summary, text)
        entities_task = loop.run_in_executor(None, extract_entities, text)
        sentiment_task = loop.run_in_executor(None, analyze_sentiment, text)

        summary, entities, sentiment = await asyncio.gather(
            summary_task, entities_task, sentiment_task
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "✅ %s processed in %.1f s  (sentiment=%s, entities=%d)",
            file_name,
            elapsed,
            sentiment,
            sum(len(v) for v in entities.values()),
        )

        # Step 3 — Build response
        return DocumentResponse(
            status="success",
            fileName=file_name,
            summary=summary,
            entities=EntitiesResponse(**entities),
            sentiment=sentiment,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error processing %s", file_name)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")


# ── Global exception handler ────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "fileName": "", "message": str(exc)},
    )
