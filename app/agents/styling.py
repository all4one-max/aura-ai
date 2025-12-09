import asyncio
from typing import Optional, List
import numpy as np

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore

from app.state import AgentState
from app.schema import ProductWithEmbedding
from app.tools.image_merging import ImageMergingService
from app.tools.embedding import EmbeddingService

import uuid


async def styling_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Visualizes products on the user by:
    1. Merging user photos with product images using Gemini 3 Pro Image Preview model
    2. Generating embeddings for each merged image using CLIP
    3. Returning list of products with embeddings appended

    For each user photo × each product, creates a ProductWithEmbedding.
    Example: 2 user photos × 10 products = 20 ProductWithEmbedding objects
    """
    print("Styling Agent")

    # Get search results from research agent
    search_results = state.get("search_results", [])
    if not search_results:
        return {
            "messages": [AIMessage(content="I couldn't find any items to style.")],
            "current_agent": "styling_agent",
            "next_step": None,
            "styled_products": [],
        }

    # Get user photos from profile
    user_profile = ["xx"]
    user_photo_urls = ["xx"]

    # if not user_photo_urls:
    #     return {
    #         "messages": [
    #             AIMessage(
    #                 content="I need your photos to show how the products look on you. Please upload your photos."
    #             )
    #         ],
    #         "current_agent": "styling_agent",
    #         "next_step": None,
    #         "styled_products": [],
    #     }

    # Initialize services
    image_merging_service = ImageMergingService()
    # embedding_service = EmbeddingService()

    # Process each user photo × each product combination
    styled_products: List[ProductWithEmbedding] = []
    processed_count = 0

    all_merged_images = []

    async def process_single_merge(user_photo_url, product):
        try:
            # Merge user photo with product image using Gemini 3 Pro Image Preview
            merged_image = await image_merging_service.merge_images(
                user_photo_url, product.image
            )
            merged_image.save(f"merged_image-{uuid.uuid4().hex}.jpg")
            return merged_image
        except Exception as e:
            print(
                f"Error processing product {product.title} with user photo {user_photo_url}: {e}"
            )
            return None

    # Create tasks for all combinations
    tasks = []
    for user_photo_url in user_photo_urls:
        for product in search_results:
            tasks.append(process_single_merge(user_photo_url, product))

    # Run tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    all_merged_images = [img for img in results if img is not None]

    return {
        "messages": [AIMessage(content="Here are your product recommendations!")],
        "current_agent": "styling_agent",
        "next_step": None,
        "merged_images": all_merged_images,
    }

    # if not styled_products:
    #     return {
    #         "messages": [
    #             AIMessage(
    #                 content="I encountered an error while processing the images. Please try again."
    #             )
    #         ],
    #         "current_agent": "styling_agent",
    #         "next_step": None,
    #         "styled_products": [],
    #     }

    # total_combinations = len(user_photo_urls) * len(search_results)
    # return {
    #     "messages": [
    #         AIMessage(
    #             content=f"I've generated styling visualizations for {processed_count} product-photo combination(s) out of {total_combinations} possible. Each product now has an embedding for sorting and matching."
    #         )
    #     ],
    #     "selected_item": search_results[0] if search_results else None,
    #     "current_agent": "styling_agent",
    #     "next_step": None,
    #     "styled_products": styled_products,
    # }
