from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.schema import ProductType


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
    product_type: Optional[ProductType] = Field(
        description=(
            "The type of clothing or fashion product the user is looking for. "
            "Must be one of: 'shirt', 'pant', or 'boots'. "
            "Map similar terms: 'top'/'blouse'/'tee' -> 'shirt', 'trousers'/'jeans' -> 'pant', "
            "'shoes'/'footwear' -> 'boots'. "
            "If the product type is not clearly a shirt, pant, or boots, leave as None."
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
    color: Optional[str] = Field(
        description=(
            "The preferred color or color preference mentioned by the user. "
            "Extract the specific color name. "
            "Examples: 'blue', 'red', 'black', 'white', 'navy', 'pastel pink', 'dark green'. "
            "Include color modifiers if mentioned (e.g., 'light blue', 'dark red'). "
            "If no color preference is mentioned, leave as None."
        ),
        default=None,
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
