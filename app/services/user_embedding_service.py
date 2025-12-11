"""
User embedding service for generating embeddings from user photos.
Creates multiple embeddings to capture different aspects of user preferences.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from PIL import Image
import asyncio
from io import BytesIO
import requests

from app.tools.embedding import EmbeddingService
from app.schema import UserEmbedding
from app.services.s3_service import s3_service


class UserEmbeddingService:
    """
    Generates user embeddings from photos for personalization.
    Creates multiple embeddings: style, color, and uses style as face/body embedding.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()

    async def _download_image(self, image_url: str) -> Image.Image:
        """
        Download image from URL or S3 key.
        If it's an S3 presigned URL, extracts the S3 key and downloads directly from S3.
        Otherwise, downloads via HTTP.
        """
        def _download():
            try:
                # Check if it's an S3 presigned URL (contains s3.amazonaws.com)
                # Extract S3 key from URL if it's a presigned URL
                s3_key = None
                if "s3.amazonaws.com" in image_url or ("s3." in image_url and ".amazonaws.com" in image_url):
                    # Extract S3 key from presigned URL
                    # Format: https://s3...amazonaws.com/bucket-name/users/username/profile/filename.jpg?query_params
                    # Or: https://bucket-name.s3.region.amazonaws.com/users/username/profile/filename.jpg?query_params
                    from urllib.parse import urlparse, unquote
                    parsed = urlparse(image_url)
                    path = parsed.path.lstrip('/')
                    
                    # Get bucket name from config
                    from app.config import AWS_S3_BUCKET
                    bucket_name = AWS_S3_BUCKET
                    
                    # Try to extract S3 key by removing bucket name from path
                    if bucket_name and path.startswith(bucket_name + '/'):
                        s3_key = unquote(path[len(bucket_name) + 1:])  # Remove bucket name and leading slash
                    elif '/' in path:
                        # If bucket name not in path, assume first part is bucket, rest is key
                        parts = path.split('/', 1)
                        if len(parts) > 1:
                            s3_key = unquote(parts[1])
                    
                    if s3_key:
                        print(f"📋 Extracted S3 key from URL: {s3_key}")
                    else:
                        print(f"⚠️  Could not extract S3 key from URL: {image_url}")
                
                # If we extracted an S3 key, use S3 service directly (no expiration issues)
                if s3_key:
                    try:
                        image_bytes = s3_service.get_object(s3_key)
                        return Image.open(BytesIO(image_bytes))
                    except Exception as s3_error:
                        print(f"⚠️  Failed to get image from S3, falling back to HTTP: {s3_error}")
                        # Fall back to HTTP download
                
                # If it's not an HTTP URL, assume it's an S3 key
                if not image_url.startswith("http"):
                    image_bytes = s3_service.get_object(image_url)
                    return Image.open(BytesIO(image_bytes))
                
                # Fallback: Download via HTTP (for non-S3 URLs or if S3 fails)
                response = requests.get(image_url, stream=True, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                raise ValueError(f"Failed to download image from {image_url}: {e}")

        return await asyncio.to_thread(_download)

    async def _extract_color_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract color preferences from image using histogram-based approach.
        Returns a 256-dimensional embedding representing dominant colors.
        """
        try:
            # Resize image for faster processing
            image_resized = image.resize((200, 200))
            
            # Convert to RGB array
            img_array = np.array(image_resized)
            
            # Extract color histogram (simpler than K-means)
            # Use reduced color space: 8 bins per channel = 8^3 = 512 colors
            # But we'll use 4 bins per channel = 4^3 = 64 colors for efficiency
            bins = 4
            hist, _ = np.histogramdd(
                img_array.reshape(-1, 3),
                bins=[bins, bins, bins],
                range=[(0, 255), (0, 255), (0, 255)]
            )
            
            # Flatten histogram and normalize
            color_hist = hist.flatten()
            color_hist = color_hist / (color_hist.sum() + 1e-8)  # Normalize
            
            # Pad to 256 dimensions
            embedding = np.zeros(256)
            embedding[:len(color_hist)] = color_hist
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            print(f"Error extracting color embedding: {e}")
            import traceback
            traceback.print_exc()
            # Return zero vector on failure
            return np.zeros(256)

    async def generate_user_embeddings(
        self, photo_urls: List[str]
    ) -> UserEmbedding:
        """
        Generate user embeddings from photo URLs.
        
        Uses the first photo (or best photo) to generate embeddings.
        For multiple photos, could average them or use the latest.
        
        Args:
            photo_urls: List of user photo URLs (S3 URLs or keys)
            
        Returns:
            UserEmbedding object with style, color, face, brand, and intent embeddings
        """
        if not photo_urls:
            raise ValueError("At least one photo URL is required to generate embeddings")
        
        # Use the first photo (or could select best quality photo)
        primary_photo_url = photo_urls[0]
        
        print(f"🔄 Generating user embeddings from photo: {primary_photo_url[:50]}...")
        
        # Download image
        image = await self._download_image(primary_photo_url)
        
        # Generate style embedding using CLIP (768-dim)
        # This captures overall style, appearance, and aesthetic preferences
        style_embedding = await self.embedding_service.get_image_embedding(image)
        
        # Extract color embedding (256-dim)
        color_embedding = await self._extract_color_embedding(image)
        
        # For now, use style embedding for face/body embedding
        # In future, could use face recognition model for face_embedding
        face_embedding = style_embedding.copy()  # 768-dim, same as style
        
        # For brand and intent embeddings, use style embedding as base
        # These could be enhanced with user interaction data later
        brand_embedding = style_embedding.copy()  # 768-dim
        intent_embedding = style_embedding.copy()  # 768-dim
        
        # Convert to lists for JSON serialization
        user_embedding = UserEmbedding(
            style_embedding=style_embedding.tolist(),
            brand_embedding=brand_embedding.tolist(),
            color_embedding=color_embedding.tolist(),
            intent_embedding=intent_embedding.tolist(),
            face_embedding=face_embedding.tolist(),
        )
        
        print(f"✅ Generated user embeddings: style={len(style_embedding)}D, color={len(color_embedding)}D")
        
        return user_embedding

    async def update_user_embeddings_from_photos(
        self, user_id: str, photo_urls: List[str]
    ) -> UserEmbedding:
        """
        Generate and return user embeddings from photos.
        This method is called when photos are uploaded.
        
        Args:
            user_id: User identifier
            photo_urls: List of photo URLs
            
        Returns:
            UserEmbedding object
        """
        if not photo_urls:
            raise ValueError("Cannot generate embeddings without photos")
        
        return await self.generate_user_embeddings(photo_urls)

