"""
Image merging service for combining user photos with product images.
Uses Google Gemini 3 Pro Image Preview model for virtual try-on/styling.
"""

import os
from pathlib import Path
from typing import List, Optional
from PIL import Image
import requests
from io import BytesIO
import asyncio

from google import genai
from google.oauth2.service_account import Credentials

from app.config import GOOGLE_VERTEX_AI_PROJECT_ID


class ImageMergingService:
    """
    Merges user photos with product images for virtual try-on using Gemini 3 Pro.
    """

    def __init__(self):
        """
        Initialize the image merging service.
        Uses credentials from aura-ai-sa-key.json file.
        """
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        cred_path = project_root / "aura-ai-sa-key.json"
        
        if not cred_path.exists():
            raise FileNotFoundError(
                f"Credentials file not found at {cred_path}. "
                "Please ensure aura-ai-sa-key.json exists in the project root."
            )
        
        # Initialize credentials and client
        self.credentials = Credentials.from_service_account_file(
            str(cred_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        
        self.project_id = GOOGLE_VERTEX_AI_PROJECT_ID
        if not self.project_id:
            raise ValueError(
                "GOOGLE_VERTEX_AI_PROJECT_ID environment variable is not set. "
                "Please set it in your .env file."
            )
        
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            credentials=self.credentials,
        )
        
        self.model_name = "gemini-3-pro-image-preview"
        self.text_prompt = """You are a virtual try on assistant. You are given a user image (first image) and a product image (second image). You need to generate a realistic image of the user(first image) wearing the product(second image)."""

    async def download_image(self, image_url: str) -> Image.Image:
        """
        Download an image from URL or S3 key.
        If it's an S3 presigned URL, extracts the S3 key and downloads directly from S3.
        Otherwise, downloads via HTTP.

        Args:
            image_url: URL of the image or S3 key

        Returns:
            PIL Image object
        """
        def _download():
            try:
                # Check if it's an S3 presigned URL (contains s3.amazonaws.com)
                # Extract S3 key from URL if it's a presigned URL
                s3_key = None
                if "s3.amazonaws.com" in image_url or "s3." in image_url and ".amazonaws.com" in image_url:
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
                    from app.services.s3_service import s3_service
                    try:
                        image_bytes = s3_service.get_object(s3_key)
                        return Image.open(BytesIO(image_bytes))
                    except Exception as s3_error:
                        print(f"⚠️  Failed to get image from S3, falling back to HTTP: {s3_error}")
                        # Fall back to HTTP download
                
                # Fallback: Download via HTTP (for non-S3 URLs or if S3 fails)
                response = requests.get(image_url, stream=True, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                raise ValueError(f"Failed to download image from {image_url}: {e}")

        return await asyncio.to_thread(_download)

    async def merge_images(self, user_photo_url: str, product_image_url: str) -> Image.Image:
        """
        Merge user photo with product image using Gemini 3 Pro Image Preview model.

        Args:
            user_photo_url: S3 URL or URL of user photo
            product_image_url: URL of product image

        Returns:
            Merged PIL Image object
        """
        try:
            # Download both images
            user_image = await self.download_image(user_photo_url)
            product_image = await self.download_image(product_image_url)

            # Use Gemini 3 Pro Image Preview model for virtual try-on
            merged_image = await self._call_gemini_model(user_image, product_image)

            return merged_image

        except Exception as e:
            raise ValueError(f"Failed to merge images: {e}")

    async def _call_gemini_model(
        self, user_image: Image.Image, product_image: Image.Image
    ) -> Image.Image:
        """
        Call Gemini 3 Pro Image Preview model for virtual try-on.
        Uses google.genai client directly (same approach as try-nanobanana-pro.py).

        Args:
            user_image: User photo
            product_image: Product image

        Returns:
            Merged image from Gemini model
        """
        try:
            # Generate image using Gemini 3 Pro Image Preview
            # Contents: [user_image, product_image, text_prompt]
            contents = [user_image, product_image, self.text_prompt]
            
            # Call Gemini model (run in thread to make it async)
            def _generate():
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                
                # Extract image from response
                for part in response.parts:
                    if part.inline_data is not None:
                        # Convert Google Image to PIL Image
                        google_image = part.as_image()
                        # Google Image has an image_bytes attribute we can use directly
                        # This avoids the save() method which requires a file path
                        if hasattr(google_image, 'image_bytes'):
                            image_bytes = google_image.image_bytes
                            pil_image = Image.open(BytesIO(image_bytes))
                        else:
                            # Fallback: save to temporary file and read it
                            import tempfile
                            import os
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                tmp_path = tmp_file.name
                            try:
                                google_image.save(tmp_path)
                                pil_image = Image.open(tmp_path)
                            finally:
                                # Clean up temporary file
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                        
                        # Convert to RGB if needed (in case it's RGBA or other format)
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        return pil_image
                    elif part.text is not None:
                        # If text response, log it
                        print(f"Gemini text response: {part.text}")
                
                raise ValueError("No image data found in Gemini response")
            
            merged_image = await asyncio.to_thread(_generate)
            return merged_image

        except Exception as e:
            raise ValueError(f"Failed to call Gemini model: {e}")

    async def merge_multiple(
        self, user_photo_url: str, product_image_urls: List[str]
    ) -> List[Image.Image]:
        """
        Merge user photo with multiple product images.

        Args:
            user_photo_url: S3 URL or URL of user photo
            product_image_urls: List of product image URLs

        Returns:
            List of merged images
        """
        merged_images = []
        for product_url in product_image_urls:
            try:
                merged = await self.merge_images(user_photo_url, product_url)
                merged_images.append(merged)
            except Exception as e:
                print(f"Failed to merge with product {product_url}: {e}")
                continue
        return merged_images
