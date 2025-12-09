"""
Centralized LLM service for external API calls.
Provides unified interface for OpenAI and Google Gemini models.
"""

import os
from typing import Optional, Any, List, Union

from langchain_openai import ChatOpenAI
from google import genai
from PIL import Image


class LLMService:
    """
    Centralized service for calling external LLMs.
    Supports OpenAI (for text/structured output) and Google Gemini (for image generation).
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        openai_temperature: float = 0,
        gemini_project_id: Optional[str] = None,
    ):
        """
        Initialize LLM service.

        Args:
            openai_model: OpenAI model name (default: "gpt-4o-mini")
            openai_temperature: Temperature for OpenAI (default: 0)
            gemini_project_id: Google Cloud project ID for Gemini.
                              If None, uses GOOGLE_VERTEX_AI_PROJECT_ID env var.
        """
        self.openai_model = openai_model
        self.openai_temperature = openai_temperature
        self._openai_client = None
        self._gemini_client = None
        self._gemini_project_id = gemini_project_id or os.getenv(
            "GOOGLE_VERTEX_AI_PROJECT_ID"
        )

    def get_openai_client(self) -> ChatOpenAI:
        """
        Get or create OpenAI client (lazy initialization).

        Returns:
            ChatOpenAI client instance
        """
        if self._openai_client is None:
            self._openai_client = ChatOpenAI(
                model=self.openai_model, temperature=self.openai_temperature
            )
        return self._openai_client

    def get_gemini_client(self) -> genai.Client:
        """
        Get or create Google Gemini client (lazy initialization).

        Returns:
            genai.Client instance

        Raises:
            ValueError: If GOOGLE_VERTEX_AI_PROJECT_ID is not set
        """
        if self._gemini_client is None:
            if not self._gemini_project_id:
                raise ValueError(
                    "GOOGLE_VERTEX_AI_PROJECT_ID environment variable must be set"
                )
            self._gemini_client = genai.Client(
                vertexai=True, project=self._gemini_project_id
            )
        return self._gemini_client

    def generate_structured_output(
        self, prompt: str, output_schema: type[Any]
    ) -> Any:
        """
        Generate structured output using OpenAI with schema.

        Args:
            prompt: Input prompt/text
            output_schema: Pydantic model class for structured output

        Returns:
            Instance of output_schema with generated data
        """
        client = self.get_openai_client()
        structured_llm = client.with_structured_output(output_schema)
        return structured_llm.invoke(prompt)

    def generate_text(self, prompt: str) -> str:
        """
        Generate text using OpenAI.

        Args:
            prompt: Input prompt/text

        Returns:
            Generated text response
        """
        client = self.get_openai_client()
        response = client.invoke(prompt)
        return response.content

    def generate_image_with_gemini(
        self,
        contents: List[Union[str, Image.Image]],
        model: str = "gemini-3-pro-image-preview",
    ) -> Image.Image:
        """
        Generate image using Google Gemini model.

        Args:
            contents: List of contents (images and/or text prompts)
            model: Gemini model name (default: "gemini-3-pro-image-preview")

        Returns:
            Generated PIL Image

        Raises:
            ValueError: If no image found in response
        """
        client = self.get_gemini_client()
        response = client.models.generate_content(model=model, contents=contents)

        # Extract image from response
        for part in response.parts:
            if part.inline_data is not None:
                return part.as_image()
            elif part.text is not None:
                # Log text response if any
                print(f"Model returned text: {part.text}")

        raise ValueError("No image found in model response")


# Global singleton instance (can be overridden for testing)
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get global LLM service instance (singleton pattern).

    Returns:
        LLMService instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def set_llm_service(service: LLMService) -> None:
    """
    Set global LLM service instance (useful for testing).

    Args:
        service: LLMService instance to use
    """
    global _llm_service
    _llm_service = service


