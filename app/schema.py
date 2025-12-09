from typing import Optional, Any

import numpy as np
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, SQLModel


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
    ) -> "ProductWithEmbedding":
        """
        Create ProductWithEmbedding from Product and embedding.

        Args:
            product: Original Product instance
            embedding: Vector embedding for merged image
            user_photo_url: Optional user photo URL used

        Returns:
            ProductWithEmbedding instance
        """
        return cls(
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
