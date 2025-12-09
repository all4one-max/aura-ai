from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# Define a subset model for extraction (excluding IDs)
class ChatQueryExtraction(BaseModel):
    """
    Extracted search parameters from the user's query.
    """

    query: str = Field(
        description=(
            "The main search query/product description. "
            "Form a comprehensive search query from the user's input. "
            "Examples: 'running shoes like messi wear', 'red dress', 'laptop', 'shirt for beach wedding'. "
            "This is REQUIRED and cannot be empty."
        )
    )
    min_price: Optional[float] = Field(
        default=None,
        description=(
            "Minimum price filter (in local currency). "
            "Extract numeric value only. Examples: 1000, 500.50, 0. "
            "Must be >= 0."
        ),
    )
    max_price: Optional[float] = Field(
        default=None,
        description=(
            "Maximum price filter (in local currency). "
            "Extract numeric value only. Examples: 5000, 10000.99. "
            "Must be >= min_price if both are set."
        ),
    )
    min_rating: Optional[float] = Field(
        default=None,
        description=(
            "Minimum product rating filter (0-5 scale). "
            "Examples: 3.0, 3.5, 4.0, 4.5, 4.7, 5.0. "
            "Must be between 0.0 and 5.0."
        ),
    )
    sort: Optional[str] = Field(
        default="relevance",
        description=(
            "Sort order for results. "
            "Must be one of: 'relevance', 'price_low', 'price_high', 'rating_high'. "
            "Default is 'relevance'."
        ),
    )
    brand: Optional[str] = Field(
        default=None,
        description=(
            "Brand name filter. "
            "Examples: 'Nike', 'Adidas', 'Apple', 'Samsung', 'Zara', 'H&M'. "
            "Case-insensitive."
        ),
    )
    color: Optional[str] = Field(
        default=None,
        description=(
            "Color filter. "
            "Examples: 'red', 'blue', 'black', 'navy', 'gold', 'striped', 'floral'. "
            "Case-insensitive."
        ),
    )
    material: Optional[str] = Field(
        default=None,
        description=(
            "Material filter. "
            "Examples: 'cotton', 'leather', 'silk', 'wool', 'denim', 'polyester'. "
            "Case-insensitive."
        ),
    )
    size: Optional[str] = Field(
        default=None,
        description=(
            "Size filter (varies by product type). "
            "Examples: 'M', 'XL', '10', '42', '28', '2T'. "
            "Format depends on product type."
        ),
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Product category/type filter. "
            "Examples: 'shoe', 'pant', 'shirt', 'sunglass', 'dress', 'jacket', 'laptop'. "
            "Handles plural/singular automatically."
        ),
    )
    store: Optional[str] = Field(
        default=None,
        description=(
            "Store/Merchant name filter. "
            "Examples: 'Myntra', 'Amazon', 'Flipkart', 'Zara', 'H&M'. "
            "Case-insensitive."
        ),
    )
    gender: Optional[str] = Field(
        default=None,
        description=(
            "Gender filter. "
            "Examples: 'Men', 'Women', 'Unisex', 'Boys', 'Girls'. "
            "Case-insensitive."
        ),
    )
    age_group: Optional[str] = Field(
        default=None,
        description=(
            "Age group filter. "
            "Examples: 'Kids', 'Toddler', 'Infant', 'Adult', 'Teen'. "
            "Case-insensitive."
        ),
    )
    condition: Optional[str] = Field(
        default=None,
        description=(
            "Product condition filter. "
            "Must be exactly 'new' or 'used' (case-insensitive). "
            "Examples: 'new', 'used'."
        ),
    )
    on_sale: bool = Field(
        default=False,
        description="Filter for items on sale. Boolean only."
    )
    free_shipping: bool = Field(
        default=False,
        description="Filter for free shipping. Boolean only."
    )
    google_domain: str = Field(
        default="google.co.in",
        description="Google domain to use. Examples: 'google.co.in', 'google.com', 'google.co.uk'."
    )
    gl: str = Field(
        default="in",
        description="Country code (ISO 3166-1 alpha-2). Examples: 'in', 'us', 'uk', 'ca', 'au'."
    )
    hl: str = Field(
        default="en",
        description="Language code (ISO 639-1). Examples: 'en', 'hi', 'es', 'fr', 'de'."
    )
    location: str = Field(
        default="India",
        description="Location string for search. Examples: 'India', 'United States', 'Mumbai, Maharashtra, India'."
    )
    start: Optional[int] = Field(
        default=None,
        description="Pagination offset (starting position). Examples: 0, 20, 40, 60. Must be >= 0."
    )
    num: Optional[int] = Field(
        default=None,
        description="Number of results per page. Examples: 10, 20, 40, 60, 100. Must be > 0."
    )
    device: Optional[str] = Field(
        default=None,
        description="Device type for search. Must be exactly 'desktop' or 'mobile'."
    )
    no_cache: bool = Field(
        default=False,
        description="Bypass cached results. Boolean only."
    )
    use_light_api: bool = Field(
        default=False,
        description="Use Google Shopping Light API instead of full API. Boolean only."
    )


def extract_chat_query_tool(user_query: str) -> ChatQueryExtraction:
    """
    Uses an LLM to extract structured query parameters from natural language input.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Use with_structured_output to force the LLM to return our schema
    structured_llm = llm.with_structured_output(ChatQueryExtraction)

    result = structured_llm.invoke(user_query)
    return result
