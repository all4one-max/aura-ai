from typing import Any, List, Optional, TypedDict

from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from app.schema import ChatQuery, Product, ProductWithEmbedding, UserEmbedding


class UserProfile(TypedDict):
    user_id: str  # User identifier
    username: str  # Username
    photo_urls: Optional[List[str]]  # S3 URLs of user photos
    user_embeddings: Optional[UserEmbedding]  # User embedding profile for ranking
    upper_body_size: Optional[str]  # User's preferred upper body size (e.g., "M", "L", "XL")
    lower_body_size: Optional[str]  # User's preferred lower body size (e.g., "M", "L", "XL")
    region: Optional[str]  # User's region/country code (e.g., "IN", "US")
    gender: Optional[str]  # User's gender (e.g., "male", "female", "other")
    age_group: Optional[str]  # User's age group (e.g., "adult", "teen", "senior")
    query_filters: Optional[dict]  # User's query filter preferences
    liked_items: Optional[List[str]]  # List of liked image/product IDs


class AgentState(TypedDict):
    request_id: Optional[str]  # Unique identifier for each request within a chat/thread
    user_profile: Optional[UserProfile]
    thread_id: str  # Chat/conversation identifier (same as chat_id)
    messages: Annotated[List[Any], add_messages]
    selected_item: Optional[Product]
    next_step: Optional[str]
    user_intent: Optional[str]  # recommendation, styling, fulfillment, general_chat
    current_agent: Optional[str]
    chat_query_json: Optional[ChatQuery]
    search_results: List[Product]
    styled_products: Optional[
        List[ProductWithEmbedding]
    ]  # Products with embeddings for merged images
    ranked_products: Optional[
        List[ProductWithEmbedding]
    ]  # Ranked products after ranking agent