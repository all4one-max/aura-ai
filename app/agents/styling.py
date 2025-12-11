import asyncio
import random
from typing import Optional, List
import numpy as np
from io import BytesIO

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore

from app.state import AgentState
from app.schema import ProductWithEmbedding, Product
from app.tools.image_merging import ImageMergingService
from app.tools.embedding import EmbeddingService
from app.services.s3_service import s3_service
from app.dao.product_embedding_dao import create_product_embedding

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
    user_profile = state.get("user_profile")
    user_photo_urls = []
    
    print(f"🔍 Styling agent - user_profile: {user_profile is not None}")
    
    # If user_profile exists, try to get photo_urls from it
    if user_profile:
        photo_urls = user_profile.get("photo_urls")
        print(f"🔍 photo_urls type: {type(photo_urls)}, value: {photo_urls}")
        if photo_urls:
            # Handle both list and other formats
            if isinstance(photo_urls, list):
                user_photo_urls = photo_urls
            elif isinstance(photo_urls, str):
                # Try to parse JSON string
                import json
                try:
                    user_photo_urls = json.loads(photo_urls)
                except:
                    user_photo_urls = []
    
    # If no photos found in user_profile, reload from User table using user_id
    if not user_photo_urls and user_profile:
        user_id = user_profile.get("user_id")
        if user_id:
            print(f"🔄 No photos in user_profile, reloading from User table for user_id: {user_id}")
            from app.dao.user_dao import get_user, user_to_profile
            user_record = await get_user(user_id=user_id)
            if user_record:
                # Reload full user profile
                updated_profile = user_to_profile(user_record)
                photo_urls = updated_profile.get("photo_urls", [])
                if photo_urls:
                    if isinstance(photo_urls, list):
                        user_photo_urls = photo_urls
                    elif isinstance(photo_urls, str):
                        import json
                        try:
                            user_photo_urls = json.loads(photo_urls)
                        except:
                            user_photo_urls = []
                    print(f"✅ Reloaded {len(user_photo_urls)} photos from User table")
                    # Update user_profile in state for future use
                    user_profile = updated_profile
                else:
                    print(f"⚠️  User {user_id} has no photos in database")
            else:
                print(f"⚠️  User {user_id} not found in database")
    
    print(f"🔍 user_photo_urls count: {len(user_photo_urls)}")
    
    if not user_photo_urls:
        print(f"⚠️  No user photos found. Asking user to upload.")
        return {
            "messages": [
                AIMessage(
                    content="I need your photos to show how the products look on you. Please upload your photos."
                )
            ],
            "current_agent": "styling_agent",
            "next_step": None,
            "styled_products": [],
            "merged_images": [],
        }

    # Select random user photo (to save costs, only merge one photo)
    selected_user_photo = random.choice(user_photo_urls)
    print(f"Selected random user photo: {selected_user_photo[:50]}...")

    # Select 2 products (first 2 products) to merge
    selected_products = search_results[:2]
    print(f"Selected {len(selected_products)} products for merging:")
    for i, product in enumerate(selected_products, 1):
        print(f"  {i}. {product.title if hasattr(product, 'title') else 'Product'}")

    # Initialize services
    image_merging_service = ImageMergingService()
    embedding_service = EmbeddingService()
    
    # Get user_id from profile
    user_id = user_profile.get("user_id") if user_profile else None
    if not user_id:
        return {
            "messages": [AIMessage(content="User profile not found. Please login again.")],
            "current_agent": "styling_agent",
            "next_step": None,
            "styled_products": [],
        }

    # Merge 2 products: random user photo × first 2 products (in parallel)
    async def process_product(product_idx: int, product: Product) -> Optional[ProductWithEmbedding]:
        """Process a single product: merge, generate embedding, upload to S3, and store in DB."""
        try:
            print(f"🔄 Processing product {product_idx + 1}/{len(selected_products)} in parallel...")
            
            # Merge user photo with product image (Google Gemini call)
            merged_image = await image_merging_service.merge_images(
                selected_user_photo, product.image
            )
            
            # Generate embedding for merged image
            embedding = await embedding_service.get_image_embedding(merged_image)
            
            # Upload merged image to S3
            merged_image_id = uuid.uuid4().hex
            s3_key = f"users/{user_id}/merged_images/{merged_image_id}.jpg"
            
            # Convert PIL Image to bytes
            image_bytes = BytesIO()
            merged_image.save(image_bytes, format="JPEG")
            image_bytes.seek(0)
            
            # Upload to S3
            s3_service.upload_image(image_bytes.read(), s3_key, content_type="image/jpeg")
            
            # Generate presigned URL for the merged image
            merged_image_url = s3_service.get_merged_image_url(s3_key)
            
            print(f"✅ Product {product_idx + 1} merged and uploaded to S3: {s3_key}")
            
            # Create ProductWithEmbedding
            product_id = f"prod_{merged_image_id}"
            product_with_embedding = ProductWithEmbedding.from_product(
                product=product,
                embedding=embedding,
                user_photo_url=selected_user_photo,
                product_id=product_id,
            )
            product_with_embedding.merged_image_url = merged_image_url
            
            # Store ProductEmbedding in database
            try:
                await create_product_embedding(
                    product=product_with_embedding,
                    user_id=user_id,
                    merged_image_s3_key=s3_key,
                    merged_image_url=merged_image_url,
                )
                print(f"✅ ProductEmbedding {product_idx + 1} stored in database")
            except Exception as db_error:
                print(f"⚠️ Warning: Failed to store ProductEmbedding {product_idx + 1} in database: {db_error}")
                import traceback
                traceback.print_exc()
            
            return product_with_embedding
            
        except Exception as e:
            print(f"❌ Error processing product {product_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Process all products in parallel
    print(f"🚀 Starting parallel processing of {len(selected_products)} products...")
    tasks = [
        process_product(product_idx, product) 
        for product_idx, product in enumerate(selected_products)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    styled_products = []
    merged_image_urls = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Task {idx + 1} raised exception: {result}")
            continue
        if result is not None:
            styled_products.append(result)
            merged_image_urls.append(result.merged_image_url)
    
    print(f"✅ Successfully processed {len(styled_products)}/{len(selected_products)} products")

    if not styled_products:
        return {
            "messages": [
                AIMessage(
                    content="I encountered an error while processing the images. Please try again."
                )
            ],
            "current_agent": "styling_agent",
            "next_step": None,
            "styled_products": [],
            "merged_images": [],
        }

    return {
        "messages": [AIMessage(content="Here are your product recommendations!")],
        "current_agent": "styling_agent",
        "next_step": "ranking_agent",  # Move to ranking agent after styling
        "styled_products": styled_products,
        "merged_images": merged_image_urls,
        "user_profile": user_profile,  # Ensure updated user_profile persists in state
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
