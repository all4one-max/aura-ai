from typing import Optional, Any, List, Dict
from datetime import datetime

import numpy as np
import json
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, SQLModel, JSON, Column
from sqlalchemy import DateTime


class Product(BaseModel):
    """
    Product schema returned from Google Shopping API search.
    Required fields: image, price, link.
    Optional fields: rating, title, source, reviews.
    """

    image: str = Field(description="Product image URL (thumbnail)", min_length=1)
    price: str = Field(description="Product price", min_length=1)
    link: str = Field(description="Product link/URL", min_length=1)
    rating: Optional[float] = Field(
        default=None, description="Product rating (0-5 scale)"
    )
    title: str = Field(default="", description="Product title/name")
    source: str = Field(default="", description="Product source/store")
    reviews: Optional[int] = Field(default=None, description="Number of reviews")

    @classmethod
    def get_required_fields(cls) -> list[str]:
        """
        Returns list of required field names from the schema.
        Only fields that are truly required (image, price, link) are returned.
        """
        # Explicitly define required fields based on schema definition
        # These are fields without defaults that must be present
        return ["image", "price", "link"]

    @classmethod
    def get_api_field_mapping(cls) -> dict[str, list[str]]:
        """
        Returns mapping of Product fields to API field names.
        Used when API uses different field names.
        """
        return {
            "image": ["thumbnail", "image"],
            "link": ["link", "product_link"],
            "price": ["price"],
        }


class UserEmbedding(BaseModel):
    """
    User embedding profile for ranking and matching.
    Contains various embeddings representing user preferences.
    """

    style_embedding: Any = Field(description="User style preference embedding")
    brand_embedding: Any = Field(description="User brand preference embedding")
    color_embedding: Any = Field(description="User color preference embedding")
    intent_embedding: Any = Field(description="User conversation/intent embedding")
    face_embedding: Any = Field(description="User face embedding")


class ProductWithEmbedding(BaseModel):
    """
    Product with embedding vector for merged image.
    Used by styling agent to return products with their styling embeddings.
    """

    # Unique identifier for the product
    id: str = Field(description="Unique identifier for the product")

    # All Product fields
    image: str = Field(description="Product image URL (thumbnail)", min_length=1)
    price: str = Field(description="Product price", min_length=1)
    link: str = Field(description="Product link/URL", min_length=1)
    rating: Optional[float] = Field(
        default=None, description="Product rating (0-5 scale)"
    )
    title: str = Field(default="", description="Product title/name")
    source: str = Field(default="", description="Product source/store")
    reviews: Optional[int] = Field(default=None, description="Number of reviews")

    # Additional fields for styling
    embedding: Any = Field(
        description="Vector embedding (768-dim) for merged image (user + product)"
    )
    user_photo_url: Optional[str] = Field(
        default=None, description="User photo URL used for merging"
    )
    merged_image_url: Optional[str] = Field(
        default=None, description="URL of merged image (if saved)"
    )

    @classmethod
    def from_product(
        cls,
        product: Product,
        embedding: np.ndarray,
        user_photo_url: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> "ProductWithEmbedding":
        """
        Create ProductWithEmbedding from Product and embedding.

        Args:
            product: Original Product instance
            embedding: Vector embedding for merged image
            user_photo_url: Optional user photo URL used
            product_id: Optional unique ID. If not provided, generates one.

        Returns:
            ProductWithEmbedding instance
        """
        import uuid
        if product_id is None:
            product_id = f"prod_{uuid.uuid4().hex[:12]}"
        
        return cls(
            id=product_id,
            image=product.image,
            price=product.price,
            link=product.link,
            rating=product.rating,
            title=product.title,
            source=product.source,
            reviews=product.reviews,
            embedding=embedding.tolist()
            if isinstance(embedding, np.ndarray)
            else embedding,
            user_photo_url=user_photo_url,
        )


class ChatQuery(SQLModel, table=True):
    """
    Structured query extracted from user input for product search.
    """

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(description="The user identifier.")
    thread_id: str = SQLField(description="The conversation thread identifier.")

    destination: Optional[str] = Field(
        description="The destination if mentioned, or general location context."
    )
    occasion: Optional[str] = Field(
        description="The occasion or event (e.g., beach wedding, business meeting)."
    )
    budget_range: Optional[str] = Field(
        description="The budget range if specified (e.g., $50-$100).", default=None
    )
    month_of_visit: Optional[str] = Field(
        description="The month of visit if relevant for season.", default=None
    )
    query: str = SQLField(
        description="The main search query/product description you're looking for"
    )

    # Optional filters
    min_price: Optional[float] = SQLField(
        default=None, description="Minimum price filter (in local currency)"
    )
    max_price: Optional[float] = SQLField(
        default=None, description="Maximum price filter (in local currency)"
    )
    min_rating: Optional[float] = SQLField(
        default=None, description="Minimum product rating filter (0-5 scale)"
    )
    sort: Optional[str] = SQLField(
        default="relevance",
        description="Sort order: relevance, price_low, price_high, rating_high",
    )
    brand: Optional[str] = SQLField(default=None, description="Brand name filter")
    color: Optional[str] = SQLField(default=None, description="Color filter")
    material: Optional[str] = SQLField(default=None, description="Material filter")
    size: Optional[str] = SQLField(default=None, description="Size filter")
    category: Optional[str] = SQLField(
        default=None, description="Product category/type filter"
    )
    store: Optional[str] = SQLField(
        default=None, description="Store/Merchant name filter"
    )
    gender: Optional[str] = SQLField(default=None, description="Gender filter")
    age_group: Optional[str] = SQLField(default=None, description="Age group filter")
    condition: Optional[str] = SQLField(
        default=None, description="Product condition: new or used"
    )
    on_sale: bool = SQLField(default=False, description="Filter for items on sale")
    free_shipping: bool = SQLField(
        default=False, description="Filter for free shipping"
    )
    google_domain: str = SQLField(
        default="google.co.in", description="Google domain to use"
    )
    gl: str = SQLField(default="in", description="Country code (ISO 3166-1 alpha-2)")
    hl: str = SQLField(default="en", description="Language code (ISO 639-1)")
    location: str = SQLField(default="India", description="Location string for search")
    start: Optional[int] = SQLField(default=None, description="Pagination offset")
    num: Optional[int] = SQLField(
        default=None, description="Number of results per page"
    )
    device: Optional[str] = SQLField(
        default=None, description="Device type: desktop or mobile"
    )
    no_cache: bool = SQLField(default=False, description="Bypass cached results")
    use_light_api: bool = SQLField(
        default=False, description="Use Google Shopping Light API"
    )


class User(SQLModel, table=True):
    """
    User SQLModel for PostgreSQL persistence.
    
    NOTE: UserProfile (in app/state.py) is the source of truth TypedDict used in AgentState.
    This User SQLModel is ONLY for PostgreSQL storage - it mirrors UserProfile structure
    but stores complex types (lists, dicts, UserEmbedding) as JSON strings.
    
    The relationship:
    - UserProfile (TypedDict) = Runtime data structure in AgentState
    - User (SQLModel) = Database persistence layer
    - Conversion happens via user_dao.user_to_profile() and update_user_profile()
    """

    user_id: str = SQLField(primary_key=True, description="User identifier")
    username: str = SQLField(unique=True, index=True, description="Username")
    photo_urls: Optional[str] = SQLField(
        default=None, description="JSON array of S3 URLs of user photos"
    )
    user_embeddings: Optional[str] = SQLField(
        default=None, description="JSON object of user embeddings"
    )
    upper_body_size: Optional[str] = SQLField(
        default=None, description="User's preferred upper body size (e.g., 'M', 'L', 'XL')"
    )
    lower_body_size: Optional[str] = SQLField(
        default=None, description="User's preferred lower body size (e.g., 'M', 'L', 'XL')"
    )
    region: Optional[str] = SQLField(
        default=None, description="User's region/country code (e.g., 'IN', 'US')"
    )
    gender: Optional[str] = SQLField(
        default=None, description="User's gender (e.g., 'male', 'female', 'other')"
    )
    age_group: Optional[str] = SQLField(
        default=None, description="User's age group (e.g., 'adult', 'teen', 'senior')"
    )
    query_filters: Optional[str] = SQLField(
        default=None, description="JSON object of user's query filter preferences"
    )
    liked_items: Optional[str] = SQLField(
        default=None, description="JSON array of liked image/product IDs"
    )


class UserChat(SQLModel, table=True):
    """
    UserChat SQLModel for tracking user-chat relationships.
    Stores which chats belong to which users.
    """
    __tablename__ = "user_chat"
    
    username: str = SQLField(index=True, description="Username")
    chat_room_id: str = SQLField(primary_key=True, index=True, description="Chat room/thread identifier (unique per chat)")
    user_id: str = SQLField(index=True, description="User identifier")
    created_at: Optional[datetime] = SQLField(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="Timestamp when chat was created"
    )
    updated_at: Optional[datetime] = SQLField(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="Timestamp when chat was last updated"
    )


class AgentStateTable(SQLModel, table=True):
    """
    Agent state table - stores AgentState fields in separate columns.
    
    One row per thread_id (chat). Updated incrementally as each agent completes:
    - context_agent completes → updates chat_query_json
    - research_agent completes → updates search_results
    - styling_agent completes → updates styled_products and merged_images
    - ranking_agent completes → updates ranked_products
    
    Note: selected_item is for future use (when user selects a specific product)
    """
    __tablename__ = "agent_state"
    
    thread_id: str = SQLField(primary_key=True, description="Chat/conversation identifier (same as chat_id)")
    user_id: str = SQLField(index=True, description="User identifier")
    request_id: Optional[str] = SQLField(default=None, description="Unique identifier for each request within a chat/thread")
    
    # AgentState fields stored in separate columns
    messages: Optional[dict] = SQLField(
        default=None,
        sa_column=Column(JSON),
        description="List of chat messages"
    )
    user_profile: Optional[dict] = SQLField(
        default=None,
        sa_column=Column(JSON),
        description="User profile data"
    )
    search_results: Optional[dict] = SQLField(
        default=None,
        sa_column=Column(JSON),
        description="List of products from research_agent (stored when research completes)"
    )
    selected_item: Optional[dict] = SQLField(
        default=None,
        sa_column=Column(JSON),
        description="Currently selected product (for future use - when user clicks/selects a product)"
    )
    chat_query_json: Optional[dict] = SQLField(
        default=None,
        sa_column=Column(JSON),
        description="Extracted chat query from context_agent (stored when context completes)"
    )
    styled_products: Optional[dict] = SQLField(
        default=None,
        sa_column=Column(JSON),
        description="List of products with embeddings from styling_agent (stored when styling completes)"
    )
    ranked_products: Optional[dict] = SQLField(
        default=None,
        sa_column=Column(JSON),
        description="List of ranked products from ranking_agent (stored when ranking completes)"
    )
    merged_images: Optional[dict] = SQLField(
        default=None,
        sa_column=Column(JSON),
        description="List of merged image URLs from styling_agent"
    )
    
    # Common fields extracted for easy querying
    current_agent: Optional[str] = SQLField(default=None, description="Current agent name")
    user_intent: Optional[str] = SQLField(default=None, description="User intent: recommendation, styling, fulfillment, general_chat")
    next_step: Optional[str] = SQLField(default=None, description="Next step in the agent flow")
    
    # Timestamps
    created_at: Optional[datetime] = SQLField(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="Timestamp when state was first created"
    )
    updated_at: Optional[datetime] = SQLField(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
        description="Timestamp when state was last updated"
    )


class ProductEmbedding(SQLModel, table=True):
    """
    Product with embedding data stored in database.
    Stores products that have been merged with user photos, along with their embeddings.
    """
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    product_id: str = SQLField(index=True, description="Unique identifier for the product")
    user_id: str = SQLField(index=True, description="User identifier who this product was shown to")
    
    # Product fields
    image: str = SQLField(description="Product image URL (thumbnail)")
    price: str = SQLField(description="Product price")
    link: str = SQLField(description="Product link/URL")
    rating: Optional[float] = SQLField(default=None, description="Product rating (0-5 scale)")
    title: str = SQLField(default="", description="Product title/name")
    source: str = SQLField(default="", description="Product source/store")
    reviews: Optional[int] = SQLField(default=None, description="Number of reviews")
    
    # Embedding and merged image data
    embedding: str = SQLField(description="JSON array of embedding vector (768-dim) for merged image")
    user_photo_url: Optional[str] = SQLField(default=None, description="User photo URL used for merging")
    merged_image_url: Optional[str] = SQLField(default=None, description="S3 URL of merged image")
    merged_image_s3_key: Optional[str] = SQLField(default=None, description="S3 key of merged image")
    
    # Metadata
    created_at: Optional[str] = SQLField(default=None, description="Timestamp when product was created")
