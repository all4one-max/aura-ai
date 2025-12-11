"""
API request and response models for user-related endpoints.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class LoginRequest(BaseModel):
    """Request model for login endpoint."""
    username: str

    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe"
            }
        }


class LoginResponse(BaseModel):
    """Response model for login endpoint."""
    user_id: str
    username: str
    profile: Dict[str, Any]  # UserProfile as dict (matches UserProfile TypedDict structure)
    query_filters: Optional[Dict[str, Any]] = None  # Top-level for frontend convenience
    embeddings: Optional[Dict[str, Any]] = None  # Top-level for frontend convenience (user_embeddings from profile)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "username": "john_doe",
                "profile": {
                    "user_id": "user_123",
                    "username": "john_doe",
                    "photo_urls": None,
                    "user_embeddings": None,
                    "upper_body_size": None,
                    "lower_body_size": None,
                    "region": None,
                    "gender": None,
                    "age_group": None,
                    "query_filters": None,
                    "liked_items": []
                },
                "query_filters": {},
                "embeddings": {}
            }
        }


class UploadUrlResponse(BaseModel):
    """Response model for upload-url endpoint."""
    upload_url: str
    image_url: str
    s3_key: str
    expires_in: int

    class Config:
        json_schema_extra = {
            "example": {
                "upload_url": "https://bucket.s3.amazonaws.com/...",
                "image_url": "https://bucket.s3.amazonaws.com/...",
                "s3_key": "users/john_doe/profile/uuid.jpg",
                "expires_in": 3600
            }
        }


class ImageUrlResponse(BaseModel):
    """Response model for image-url endpoint."""
    image_url: str
    s3_key: str
    expires_in: int

    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://bucket.s3.amazonaws.com/...",
                "s3_key": "users/john_doe/profile/uuid.jpg",
                "expires_in": 3600
            }
        }


class UpdateRequest(BaseModel):
    """Request model for update endpoint. Accepts any UserProfile fields."""
    photo_urls: Optional[List[str]] = None
    upper_body_size: Optional[str] = None
    lower_body_size: Optional[str] = None
    region: Optional[str] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None
    query_filters: Optional[Dict[str, Any]] = None
    liked_items: Optional[List[str]] = None
    s3_key: Optional[str] = None  # For image uploads - converts to photo_urls
    
    class Config:
        json_schema_extra = {
            "example": {
                "upper_body_size": "M",
                "lower_body_size": "L",
                "region": "IN",
                "gender": "male",
                "age_group": "adult",
                "s3_key": "users/john_doe/profile/uuid.jpg"
            }
        }


class UpdateResponse(BaseModel):
    """Response model for update endpoint."""
    user_id: str
    username: str
    profile: Dict[str, Any]
    query_filters: Optional[Dict[str, Any]] = None
    embeddings: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None


class LikeResponse(BaseModel):
    """Response model for like endpoint."""
    success: bool
    message: str
    liked_items: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Item liked successfully",
                "liked_items": ["image123", "image456"]
            }
        }


class CreateChatRequest(BaseModel):
    """Request model for createChat endpoint."""
    user_id: str  # User ID from frontend (e.g., "user_123" or actual user_id)
    session_name: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "session_name": "Shopping Session 1"
            }
        }


class CreateChatResponse(BaseModel):
    """Response model for createChat endpoint."""
    id: str
    chat_id: str
    username: str
    session_name: Optional[str] = None
    messages: List[Dict[str, Any]] = []

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chat_123",
                "chat_id": "chat_123",
                "username": "john_doe",
                "session_name": "Shopping Session 1",
                "messages": []
            }
        }


class ChatInfo(BaseModel):
    """Chat information model."""
    id: str
    chat_id: str
    username: str
    session_name: Optional[str] = None
    messages: List[Dict[str, Any]] = []

