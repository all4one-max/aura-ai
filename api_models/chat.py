"""
API request and response models for Aura AI endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str
    user_id: str
    thread_id: Optional[str] = None  # Auto-generate if not provided

    class Config:
        json_schema_extra = {
            "example": {
                "message": "I need a hat for a beach party in Thailand",
                "user_id": "user123",
                "thread_id": "thread456",
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    thread_id: str
    user_id: str
    request_id: str  # Unique identifier for this request
    merged_images: Optional[List[str]] = None
    styled_products: Optional[List[dict]] = None  # Deprecated: Use ranked_products instead (kept for backward compatibility)
    ranked_products: Optional[List[dict]] = None  # Ranked products with embeddings and merged image URLs (prioritized)

    class Config:
        json_schema_extra = {
            "example": {
                "response": "I'd be happy to help you find a hat! What occasion is this for?",
                "thread_id": "thread456",
                "user_id": "user123",
                "current_agent": "context_agent",
                "next_step": "clarification_agent",
            }
        }
