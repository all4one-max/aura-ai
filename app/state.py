from typing import Any, List, Optional, TypedDict

from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from app.schema import ChatQuery, Product, ProductWithEmbedding, UserEmbedding


class UserProfile(TypedDict):
    body_type: Optional[str]
    skin_tone: Optional[str]
    preferences: List[str]
    sizes: Optional[dict]
    photo_urls: Optional[List[str]]  # S3 URLs of user photos
    user_embeddings: Optional[UserEmbedding]  # User embedding profile for ranking


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_profile: Optional[UserProfile]
    search_results: List[Product]
    selected_item: Optional[Product]
    next_step: Optional[str]
    user_intent: Optional[str]  # recommendation, styling, fulfillment, general_chat
    current_agent: Optional[str]
    thread_id: int
    chat_query_json: Optional[ChatQuery]
    styled_products: Optional[List[ProductWithEmbedding]]  # Products with embeddings for merged images
    ranked_products: Optional[List[ProductWithEmbedding]]  # Ranked products after ranking agent
