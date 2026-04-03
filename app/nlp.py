"""
NLP analysis pipeline.

- Summarization   → Groq API (llama-3.3-70b-versatile, free tier)
- NER             → spaCy (en_core_web_sm, local)
- Sentiment       → Groq API (same model, free tier)

Using Groq instead of local HuggingFace models keeps RAM usage minimal
(critical for Render.com free tier's 512 MB limit) while providing
high-quality results at zero cost.
"""

from __future__ import annotations

import logging
from typing import Any

import spacy
from groq import Groq

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Lazy-loaded singletons ───────────────────────────────────────────────────

_nlp_spacy: spacy.Language | None = None
_groq_client: Groq | None = None


def _get_spacy() -> spacy.Language:
    """Load spaCy model once (with only NER enabled for speed)."""
    global _nlp_spacy
    if _nlp_spacy is None:
        settings = get_settings()
        logger.info("Loading spaCy model: %s", settings.SPACY_MODEL)
        _nlp_spacy = spacy.load(
            settings.SPACY_MODEL,
            disable=["parser", "lemmatizer"],  # We only need NER
        )
        logger.info("spaCy model loaded successfully")
    return _nlp_spacy


def _get_groq() -> Groq:
    """Create Groq client once."""
    global _groq_client
    if _groq_client is None:
        settings = get_settings()
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
            )
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info("Groq client initialized")
    return _groq_client


# ── Summarization (Groq) ────────────────────────────────────────────────────


def generate_summary(text: str) -> str:
    """
    Generate a concise summary of the document text using Groq API.
    For very short texts, returns them as-is.
    """
    if len(text.strip()) < 50:
        return text.strip() if text.strip() else "No meaningful text found in the document."

    settings = get_settings()
    client = _get_groq()

    # Truncate very long texts to stay within token limits (~4 chars/token)
    max_chars = 12000
    truncated = text[:max_chars] if len(text) > max_chars else text

    try:
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise document summarizer. "
                        "Produce a concise summary of the following document in 2-4 sentences. "
                        "Capture the key facts, figures, and main topic. "
                        "Do NOT include any preamble like 'Here is a summary' — just output the summary directly."
                    ),
                },
                {
                    "role": "user",
                    "content": truncated,
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )
        summary = completion.choices[0].message.content.strip()
        logger.info("Groq summary generated (%d chars)", len(summary))
        return summary

    except Exception as exc:
        logger.error("Groq summarization failed: %s", exc)
        # Fallback: return first 500 chars as a crude summary
        return text[:500].strip() + "..."


# ── Named Entity Recognition (spaCy) ────────────────────────────────────────

_ENTITY_MAP: dict[str, str] = {
    "PERSON": "names",
    "DATE": "dates",
    "ORG": "organizations",
    "GPE": "locations",
    "LOC": "locations",
    "MONEY": "amounts",
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract named entities from text using spaCy.
    Returns dict with keys: names, dates, organizations, locations, amounts.
    """
    result: dict[str, list[str]] = {
        "names": [],
        "dates": [],
        "organizations": [],
        "locations": [],
        "amounts": [],
    }

    if not text.strip():
        return result

    nlp = _get_spacy()

    # spaCy has a max length; chunk if needed
    max_length = nlp.max_length
    chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]

    seen: set[str] = set()

    for chunk in chunks:
        doc = nlp(chunk)
        for ent in doc.ents:
            category = _ENTITY_MAP.get(ent.label_)
            if category is None:
                continue
            # Normalize whitespace and skip duplicates
            entity_text = " ".join(ent.text.split())
            if entity_text and entity_text not in seen:
                result[category].append(entity_text)
                seen.add(entity_text)

    logger.info(
        "Entities extracted — names=%d, dates=%d, orgs=%d, locs=%d, amounts=%d",
        len(result["names"]),
        len(result["dates"]),
        len(result["organizations"]),
        len(result["locations"]),
        len(result["amounts"]),
    )
    return result


# ── Sentiment Analysis (Groq) ───────────────────────────────────────────────


def analyze_sentiment(text: str) -> str:
    """
    Classify the overall sentiment of the text as Positive, Negative, or Neutral
    using Groq API.
    """
    if len(text.strip()) < 10:
        return "Neutral"

    settings = get_settings()
    client = _get_groq()

    # Use a representative sample for sentiment (first ~3000 chars)
    sample = text[:3000] if len(text) > 3000 else text

    try:
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment classifier. "
                        "Analyze the overall tone/sentiment of the following text. "
                        "Respond with EXACTLY one word: Positive, Negative, or Neutral. "
                        "Do not include any explanation or punctuation."
                    ),
                },
                {
                    "role": "user",
                    "content": sample,
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = completion.choices[0].message.content.strip()

        # Normalize the response
        sentiment = raw.capitalize()
        if sentiment not in ("Positive", "Negative", "Neutral"):
            # Try to extract the label from a longer response
            for label in ("Positive", "Negative", "Neutral"):
                if label.lower() in raw.lower():
                    sentiment = label
                    break
            else:
                sentiment = "Neutral"

        logger.info("Groq sentiment: %s (raw: %s)", sentiment, raw)
        return sentiment

    except Exception as exc:
        logger.error("Groq sentiment analysis failed: %s", exc)
        return "Neutral"


# ── Startup helper ───────────────────────────────────────────────────────────


def preload_models() -> None:
    """Pre-load spaCy model at application startup."""
    _get_spacy()
    logger.info("NLP models pre-loaded")
