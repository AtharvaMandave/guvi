"""
Pydantic models for API request/response contracts.
Uses Base64Bytes for automatic Base64 decoding.
"""

from pydantic import BaseModel, Base64Bytes, Field


# ── Request ──────────────────────────────────────────────────────────────────

class DocumentRequest(BaseModel):
    """Incoming document analysis request."""

    fileName: str = Field(
        ...,
        description="Original file name (e.g. 'report.pdf')",
        examples=["report.pdf"],
    )
    fileType: str = Field(
        ...,
        description="File extension: pdf, docx, png, jpg, jpeg, tiff",
        examples=["pdf"],
    )
    fileBase64: Base64Bytes = Field(
        ...,
        description="Base64-encoded file content",
    )


# ── Response ─────────────────────────────────────────────────────────────────

class EntitiesResponse(BaseModel):
    """Extracted named entities grouped by category."""

    names: list[str] = Field(default_factory=list, description="Person names")
    dates: list[str] = Field(default_factory=list, description="Date references")
    organizations: list[str] = Field(default_factory=list, description="Organization names")
    locations: list[str] = Field(default_factory=list, description="Locations and geo-political entities")
    amounts: list[str] = Field(default_factory=list, description="Monetary amounts")


class DocumentResponse(BaseModel):
    """Successful analysis response."""

    status: str = Field(default="success", description="Request status")
    fileName: str = Field(..., description="Original file name")
    summary: str = Field(..., description="AI-generated document summary")
    entities: EntitiesResponse = Field(
        default_factory=EntitiesResponse,
        description="Extracted named entities",
    )
    sentiment: str = Field(
        ...,
        description="Overall document sentiment: Positive, Negative, or Neutral",
    )


class ErrorResponse(BaseModel):
    """Error response."""

    status: str = Field(default="error")
    fileName: str = Field(default="")
    message: str = Field(..., description="Error description")
