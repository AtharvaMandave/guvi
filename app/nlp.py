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


from groq import Groq

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Lazy-loaded singletons ───────────────────────────────────────────────────


_groq_client: Groq | None = None



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


# ── Named Entity Recognition (Groq) ────────────────────────────────────────

import json

def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract named entities from text using Groq to explicitly filter out headings/titles.
    Returns dict with keys: names, dates, organizations, locations, amounts.
    """
    default_result: dict[str, list[str]] = {
        "names": [],
        "dates": [],
        "organizations": [],
        "locations": [],
        "amounts": [],
    }

    if len(text.strip()) < 10:
        return default_result

    settings = get_settings()
    client = _get_groq()

    # Truncate text to stay within token limits
    max_chars = 12000
    truncated = text[:max_chars] if len(text) > max_chars else text

    system_prompt = (
        "You are an expert Named Entity Recognition system.\n"
        "Extract the following entities from the text:\n"
        "- names: Specific human names ONLY. Do NOT extract job titles, roles, or software/tools (e.g. ignore 'Graphic Designer', 'Adobe').\n"
        "- dates: Specific dates, years, or time periods (e.g. 'June 2020 - Present', '2017').\n"
        "- organizations: Real-world identifiable companies, universities, brands, or agencies. Do NOT extract section headings (e.g. 'Skills', 'Interests'), job titles, or software (e.g. 'Figma') as organizations.\n"
        "- locations: Physical cities, states, or countries.\n"
        "- amounts: Monetary amounts, percentages (e.g. '30%', '25%'), or specific numerical business metrics.\n"
        "CRITICAL RULES:\n"
        "1. Do NOT extract section titles, headings, or descriptive phrases as entities.\n"
        "2. Ignore OCR artifacts or random text blocks.\n"
        "3. You must respond with ONLY raw, valid JSON containing EXACTLY the keys: "
        '["names", "dates", "organizations", "locations", "amounts"]. Do not use Markdown formatting or text.'
    )

    try:
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        raw_response = completion.choices[0].message.content.strip()
        parsed = json.loads(raw_response)
        
        # Ensure all keys exist and are lists
        result = {}
        for key in default_result:
            val = parsed.get(key, [])
            if not isinstance(val, list):
                val = [val] if val else []
            # Deduplicate items
            result[key] = list(dict.fromkeys(str(v).strip() for v in val if v))
            
        logger.info(
            "Entities (Groq) — names=%d, dates=%d, orgs=%d, locs=%d, amounts=%d",
            len(result["names"]), len(result["dates"]), len(result["organizations"]),
            len(result["locations"]), len(result["amounts"])
        )
        return result

    except Exception as exc:
        logger.error("Groq NER failed: %s", exc)
        return default_result


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
    """Pre-resolve Groq client at application startup."""
    _get_groq()
    logger.info("Groq client pre-loaded")
