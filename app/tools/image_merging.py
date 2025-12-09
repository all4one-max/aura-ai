"""
Image merging service for combining user photos with product images.
Uses Google Gemini 3 Pro Image Preview model for virtual try-on/styling.
"""

from typing import List, Optional
from PIL import Image
import requests
from io import BytesIO

from app.services.llm_service import get_llm_service


class ImageMergingService:
    """
    Merges user photos with product images for virtual try-on using Gemini 3 Pro.
    """

    def __init__(self):
        """
        Initialize the image merging service.
        """
        self.llm_service = get_llm_service()
        self.model_name = "gemini-3-pro-image-preview"
        self.text_prompt = """You are a virtual try on assistant. You are given a user image (first image) and a product image (second image). You need to generate a realistic image of the user(first image) wearing the product(second image)."""

    def download_image(self, image_url: str) -> Image.Image:
        """
        Download an image from URL.

        Args:
            image_url: URL of the image

        Returns:
            PIL Image object
        """
        try:
            response = requests.get(image_url, stream=True, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Failed to download image from {image_url}: {e}")

    def merge_images(self, user_photo_url: str, product_image_url: str) -> Image.Image:
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
            # user_image = self.download_image(user_photo_url)
            product_image = self.download_image(product_image_url)

            # Use Gemini 3 Pro Image Preview model for virtual try-on
            merged_image = self._call_gemini_model(
                Image.open("/Users/sauravjha/Downloads/sj-wayanad.png"), product_image
            )

            return merged_image

        except Exception as e:
            raise ValueError(f"Failed to merge images: {e}")

    def _call_gemini_model(
        self, user_image: Image.Image, product_image: Image.Image
    ) -> Image.Image:
        """
        Call Gemini 3 Pro Image Preview model for virtual try-on.

        Args:
            user_image: User photo
            product_image: Product image

        Returns:
            Merged image from Gemini model
        """
        try:
            # Generate image using Gemini 3 Pro Image Preview via LLM service
            # Contents: [user_image, product_image, text_prompt]
            contents = [user_image, product_image, self.text_prompt]
            merged_image = self.llm_service.generate_image_with_gemini(
                contents=contents, model=self.model_name
            )
            return merged_image

        except Exception as e:
            raise ValueError(f"Failed to call Gemini model: {e}")

    def merge_multiple(
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
                merged = self.merge_images(user_photo_url, product_url)
                merged_images.append(merged)
            except Exception as e:
                print(f"Failed to merge with product {product_url}: {e}")
                continue
        return merged_images
