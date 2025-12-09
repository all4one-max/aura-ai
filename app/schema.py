from typing import Optional

from sqlmodel import Field, SQLModel


class ChatQuery(SQLModel, table=True):
    """
    Structured query extracted from user input for product search.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(description="The user identifier.")
    thread_id: str = Field(description="The conversation thread identifier.")
    
    # Required field
    query: str = Field(description="The main search query/product description you're looking for")
    
    # Optional filters
    min_price: Optional[float] = Field(default=None, description="Minimum price filter (in local currency)")
    max_price: Optional[float] = Field(default=None, description="Maximum price filter (in local currency)")
    min_rating: Optional[float] = Field(default=None, description="Minimum product rating filter (0-5 scale)")
    sort: Optional[str] = Field(default="relevance", description="Sort order: relevance, price_low, price_high, rating_high")
    brand: Optional[str] = Field(default=None, description="Brand name filter")
    color: Optional[str] = Field(default=None, description="Color filter")
    material: Optional[str] = Field(default=None, description="Material filter")
    size: Optional[str] = Field(default=None, description="Size filter")
    category: Optional[str] = Field(default=None, description="Product category/type filter")
    store: Optional[str] = Field(default=None, description="Store/Merchant name filter")
    gender: Optional[str] = Field(default=None, description="Gender filter")
    age_group: Optional[str] = Field(default=None, description="Age group filter")
    condition: Optional[str] = Field(default=None, description="Product condition: new or used")
    on_sale: bool = Field(default=False, description="Filter for items on sale")
    free_shipping: bool = Field(default=False, description="Filter for free shipping")
    google_domain: str = Field(default="google.co.in", description="Google domain to use")
    gl: str = Field(default="in", description="Country code (ISO 3166-1 alpha-2)")
    hl: str = Field(default="en", description="Language code (ISO 639-1)")
    location: str = Field(default="India", description="Location string for search")
    start: Optional[int] = Field(default=None, description="Pagination offset")
    num: Optional[int] = Field(default=None, description="Number of results per page")
    device: Optional[str] = Field(default=None, description="Device type: desktop or mobile")
    no_cache: bool = Field(default=False, description="Bypass cached results")
    use_light_api: bool = Field(default=False, description="Use Google Shopping Light API")
