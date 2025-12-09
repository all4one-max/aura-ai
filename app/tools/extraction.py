from typing import Optional

from pydantic import BaseModel, Field

from app.services.llm_service import get_llm_service
from langchain_core.messages import HumanMessage


# Define a subset model for extraction (excluding IDs)
class ChatQueryExtraction(BaseModel):
    """
    Extracted search parameters from the user's query.
    """

    destination: Optional[str] = Field(
        description=(
            "The travel destination or location mentioned by the user. "
            "Extract the city, country, or region name. "
            "Examples: 'Paris', 'Bali', 'New York', 'Thailand', 'Hawaii', 'Italy'. "
            "If no specific location is mentioned, leave as None."
        ),
        default=None,
    )
    occasion: Optional[str] = Field(
        description=(
            "The specific event, occasion, or context for which the user needs clothing. "
            "Be specific and descriptive. "
            "Examples: 'beach wedding', 'business meeting', 'casual dinner', 'job interview', "
            "'music festival', 'beach party', 'formal gala', 'hiking trip', 'date night'. "
            "If no occasion is mentioned, leave as None."
        ),
        default=None,
    )
    budget_range: Optional[str] = Field(
        description=(
            "The price range or budget mentioned by the user. "
            "Extract as a range with currency symbol if provided. "
            "Examples: '$50-$100', '$200-$500', 'under $100', 'around $150', '€100-€200'. "
            "If no budget is mentioned, leave as None."
        ),
        default=None,
    )
    month_of_visit: Optional[str] = Field(
        description=(
            "The month when the user plans to travel or attend the event. "
            "Extract the full month name. "
            "Examples: 'January', 'February', 'March', 'July', 'December'. "
            "Also recognize phrases like 'next month', 'summer' (June/July/August), "
            "'winter' (December/January/February). "
            "If no timeframe is mentioned, leave as None."
        ),
        default=None,
    )

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
        default=False, description="Filter for items on sale. Boolean only."
    )
    free_shipping: bool = Field(
        default=False, description="Filter for free shipping. Boolean only."
    )
    google_domain: str = Field(
        default="google.co.in",
        description="Google domain to use. Examples: 'google.co.in', 'google.com', 'google.co.uk'.",
    )
    gl: str = Field(
        default="in",
        description="Country code (ISO 3166-1 alpha-2). Examples: 'in', 'us', 'uk', 'ca', 'au'.",
    )
    hl: str = Field(
        default="en",
        description="Language code (ISO 639-1). Examples: 'en', 'hi', 'es', 'fr', 'de'.",
    )
    location: str = Field(
        default="India",
        description="Location string for search. Examples: 'India', 'United States', 'Mumbai, Maharashtra, India'.",
    )
    start: Optional[int] = Field(
        default=None,
        description="Pagination offset (starting position). Examples: 0, 20, 40, 60. Must be >= 0.",
    )
    num: Optional[int] = Field(
        default=None,
        description="Number of results per page. Examples: 10, 20, 40, 60, 100. Must be > 0.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Device type for search. Must be exactly 'desktop' or 'mobile'.",
    )
    no_cache: bool = Field(
        default=False, description="Bypass cached results. Boolean only."
    )
    use_light_api: bool = Field(
        default=False,
        description="Use Google Shopping Light API instead of full API. Boolean only.",
    )


def extract_chat_query_tool(messages: list) -> ChatQueryExtraction:
    """
    Uses an LLM to extract structured query parameters from conversation messages.
    Builds a comprehensive query from last 5 user messages + extracted context fields.

    Args:
        messages: List of conversation messages (LangChain BaseMessage objects)

    Returns:
        ChatQueryExtraction with all extracted fields including constructed query
    """

    # Get last 5 user messages for context
    user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
    last_5_user_messages = (
        user_messages[-5:] if len(user_messages) > 5 else user_messages
    )

    # Combine all user messages into context
    conversation_context = " ".join(last_5_user_messages)

    # First, extract all fields from the conversation
    llm_service = get_llm_service()

    extraction_prompt = f"""Extract structured information from this conversation:

{conversation_context}

Extract all relevant fields (destination, occasion, month_of_visit, product details, filters, etc.).
For the 'query' field, create a comprehensive search query that combines:
1. The main product/item mentioned across all messages
2. The destination (if mentioned)
3. The occasion (if mentioned)  
4. The month/season (if mentioned)

Example:
Messages: "recommend me some hats" + "i am going to thailand" + "for a beach party"
Query should be: "hats for beach party in thailand"

Messages: "I need shirts" + "going to bali" + "in december" + "for a wedding"
Query should be: "shirts for wedding in bali december"

Build a natural, comprehensive search query from the conversation."""

    result = llm_service.generate_structured_output(
        extraction_prompt, ChatQueryExtraction
    )
    return result
