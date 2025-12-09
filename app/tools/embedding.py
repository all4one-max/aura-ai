"""
Image embedding service using CLIP model.
Generates vector embeddings for images.
"""

import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import List, Optional, Union


class EmbeddingService:
    """
    Generates vector embeddings for images using CLIP.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load the CLIP model to avoid heavy initialization."""
        if self._model is None:
            print(f"Loading CLIP model: {self.model_name}...")
            from transformers import CLIPProcessor, CLIPModel

            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            print("Model loaded.")

    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """
        Generates embedding for an image (from URL or PIL Image).

        Args:
            image_input: URL string of the image OR PIL Image object

        Returns:
            Normalized 768-dimensional numpy array embedding vector
        """
        self._load_model()

        try:
            # Handle both URL string and PIL Image
            if isinstance(image_input, str):
                # Download image from URL
                response = requests.get(image_input, stream=True, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif isinstance(image_input, Image.Image):
                # Use PIL Image directly
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Process image
            inputs = self._processor(images=image, return_tensors="pt")

            # Generate embedding
            outputs = self._model.get_image_features(**inputs)

            # Convert to numpy and normalize
            embedding = outputs.detach().numpy().flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            input_desc = image_input if isinstance(image_input, str) else "PIL Image"
            print(f"Error generating embedding for {input_desc}: {e}")
            # Return zero vector on failure to avoid crashing the pipeline
            return np.zeros(768)

    def get_image_embeddings(
        self, image_inputs: List[Union[str, Image.Image]]
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple images.

        Args:
            image_inputs: List of image URLs or PIL Image objects

        Returns:
            List of embedding vectors
        """
        return [self.get_image_embedding(img) for img in image_inputs]

