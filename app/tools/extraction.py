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
        description="The destination if mentioned, or general location context.",
        default=None,
    )
    occasion: Optional[str] = Field(
        description="The occasion or event (e.g., beach wedding, business meeting).",
        default=None,
    )
    budget_range: Optional[str] = Field(
        description="The budget range if specified (e.g., $50-$100).", default=None
    )
    product_type: Optional[ProductType] = Field(
        description="The type of product being searched for.", default=None
    )
    month_of_visit: Optional[str] = Field(
        description="The month of visit if relevant for season.", default=None
    )
    color: Optional[str] = Field(
        description="The preferred color of the product.", default=None
    )


def extract_chat_query_tool(user_query: str) -> ChatQueryExtraction:
    """
    Uses an LLM to extract structured query parameters from natural language input.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Use with_structured_output to force the LLM to return our schema
    structured_llm = llm.with_structured_output(ChatQueryExtraction)

    result = structured_llm.invoke(user_query)
    return result
