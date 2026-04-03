"""
Unit tests for the NLP pipeline.
These test spaCy NER locally (no API key needed).
Groq-dependent tests are skipped unless a real key is available.
"""

import os
import pytest

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")

from app.nlp import extract_entities  # noqa: E402


class TestEntityExtraction:
    """Test spaCy NER (runs locally, no API key needed)."""

    def test_extracts_person_names(self):
        text = "Steve Jobs founded Apple in Cupertino."
        entities = extract_entities(text)
        assert any("Steve Jobs" in name for name in entities["names"])

    def test_extracts_organizations(self):
        text = "Google and Microsoft are major tech companies."
        entities = extract_entities(text)
        org_text = " ".join(entities["organizations"])
        assert "Google" in org_text or "Microsoft" in org_text

    def test_extracts_dates(self):
        text = "The meeting is scheduled for January 15, 2025."
        entities = extract_entities(text)
        assert len(entities["dates"]) > 0

    def test_extracts_money(self):
        text = "The deal was worth $3.5 billion."
        entities = extract_entities(text)
        assert len(entities["amounts"]) > 0

    def test_empty_text(self):
        entities = extract_entities("")
        assert entities == {
            "names": [],
            "dates": [],
            "organizations": [],
            "locations": [],
            "amounts": [],
        }

    def test_no_entities_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        entities = extract_entities(text)
        # Should return valid structure even with no entities
        assert isinstance(entities["names"], list)
        assert isinstance(entities["dates"], list)

    def test_deduplication(self):
        text = "Apple Inc. is great. Apple Inc. makes iPhones. Apple Inc. is valued highly."
        entities = extract_entities(text)
        # "Apple Inc." should appear only once
        apple_count = sum(1 for o in entities["organizations"] if "Apple" in o)
        assert apple_count <= 1
