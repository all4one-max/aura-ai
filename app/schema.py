from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ProductType(str, Enum):
    SHIRT = "shirt"
    PANT = "pant"
    BOOTS = "boots"


class ChatQuery(SQLModel, table=True):
    """
    Structured query extracted from user input.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(description="The user identifier.")
    thread_id: str = Field(description="The conversation thread identifier.")
    destination: Optional[str] = Field(
        description="The destination if mentioned, or general location context."
    )
    occasion: Optional[str] = Field(
        description="The occasion or event (e.g., beach wedding, business meeting)."
    )
    budget_range: Optional[str] = Field(
        description="The budget range if specified (e.g., $50-$100).", default=None
    )
    product_type: Optional[ProductType] = Field(
        description="The type of product being searched for."
    )
    month_of_visit: Optional[str] = Field(
        description="The month of visit if relevant for season.", default=None
    )
    color: Optional[str] = Field(
        description="The preferred color of the product.", default=None
    )
