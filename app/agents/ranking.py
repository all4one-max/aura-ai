"""
Ranking agent for sorting styled products based on user preferences and beauty standards.
"""

from typing import Optional, List
import numpy as np

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore

from app.state import AgentState
from app.schema import ProductWithEmbedding, UserEmbedding
from app.utils.similarity import compute_cosine_similarity
from app.tools.beauty_standard import get_beauty_standard_embedding


def rank_merged_images(
    user_embedding: UserEmbedding,
    merged_image_embeddings: List[np.ndarray],
    beauty_standard_embedding: np.ndarray,
) -> List[int]:
    """
    Ranks merged images based on user preferences and beauty standards.

    Args:
        user_embedding: The user's current preference profile.
        merged_image_embeddings: List of vectors for the candidate images.
        beauty_standard_embedding: The reference vector for "beauty".

    Returns:
        List of indices of the merged_image_embeddings, sorted by rank (highest score first).
    """
    scores = []

    # Weights for the final score
    W_STYLE = 0.40
    W_BRAND = 0.20
    W_COLOR = 0.20
    W_INTENT = 0.10
    W_BEAUTY = 0.10

    for idx, img_vec in enumerate(merged_image_embeddings):
        # Convert to numpy array if needed
        if isinstance(img_vec, list):
            img_vec = np.array(img_vec)

        # Convert user embeddings to numpy arrays if needed
        style_emb = (
            np.array(user_embedding.style_embedding)
            if isinstance(user_embedding.style_embedding, list)
            else user_embedding.style_embedding
        )
        brand_emb = (
            np.array(user_embedding.brand_embedding)
            if isinstance(user_embedding.brand_embedding, list)
            else user_embedding.brand_embedding
        )
        color_emb = (
            np.array(user_embedding.color_embedding)
            if isinstance(user_embedding.color_embedding, list)
            else user_embedding.color_embedding
        )
        intent_emb = (
            np.array(user_embedding.intent_embedding)
            if isinstance(user_embedding.intent_embedding, list)
            else user_embedding.intent_embedding
        )

        # 1. Compute similarity scores for each user vector
        style_score = compute_cosine_similarity(style_emb, img_vec)
        brand_score = compute_cosine_similarity(brand_emb, img_vec)
        color_score = compute_cosine_similarity(color_emb, img_vec)
        intent_score = compute_cosine_similarity(intent_emb, img_vec)

        # 2. Compute beauty score
        beauty_score = compute_cosine_similarity(beauty_standard_embedding, img_vec)

        # 3. Combine scores
        final_score = (
            W_STYLE * style_score
            + W_BRAND * brand_score
            + W_COLOR * color_score
            + W_INTENT * intent_score
            + W_BEAUTY * beauty_score
        )

        scores.append((idx, final_score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return just the indices
    return [x[0] for x in scores]


async def ranking_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Ranks styled products based on user embeddings and beauty standards.

    Uses weighted cosine similarity scores from:
    - User style embedding (40%)
    - User brand embedding (20%)
    - User color embedding (20%)
    - User intent/conversation embedding (10%)
    - Beauty standard embedding (10%)
    """
    print("Ranking Agent")

    # Get styled products from styling agent
    styled_products = state.get("styled_products", [])
    if not styled_products:
        return {
            "messages": [AIMessage(content="No styled products to rank.")],
            "current_agent": "ranking_agent",
            "next_step": None,
            "ranked_products": [],
        }

    # Get user embeddings from profile
    user_profile = state.get("user_profile", {})
    user_embeddings_raw = user_profile.get("user_embeddings")

    if not user_embeddings_raw:
        return {
            "messages": [
                AIMessage(
                    content="User preference embeddings not found. Cannot rank products without user profile."
                )
            ],
            "current_agent": "ranking_agent",
            "next_step": None,
            "ranked_products": styled_products,  # Return unranked if no embeddings
        }

    # Convert to UserEmbedding if it's a dict
    if isinstance(user_embeddings_raw, dict):
        user_embeddings = UserEmbedding(**user_embeddings_raw)
    else:
        user_embeddings = user_embeddings_raw

    # Get beauty standard embedding from config
    # Can be overridden via config metadata if needed
    beauty_standard_embedding = config.get("metadata", {}).get(
        "beauty_standard_embedding"
    )

    if beauty_standard_embedding is None:
        # Load from config file
        beauty_standard_embedding = get_beauty_standard_embedding()
    else:
        # Convert to numpy array if provided via config
        if isinstance(beauty_standard_embedding, list):
            beauty_standard_embedding = np.array(beauty_standard_embedding)

    # Convert embeddings to numpy arrays
    merged_image_embeddings = []
    for product in styled_products:
        emb = product.embedding
        if isinstance(emb, list):
            emb = np.array(emb)
        merged_image_embeddings.append(emb)

    # Rank the products
    ranked_indices = rank_merged_images(
        user_embedding=user_embeddings,
        merged_image_embeddings=merged_image_embeddings,
        beauty_standard_embedding=beauty_standard_embedding,
    )

    # Reorder products based on ranking
    ranked_products = [styled_products[idx] for idx in ranked_indices]

    return {
        "messages": [
            AIMessage(
                content=f"I've ranked {len(ranked_products)} products based on your preferences and style. Here are the top recommendations."
            )
        ],
        "current_agent": "ranking_agent",
        "next_step": None,
        "ranked_products": ranked_products,
    }
