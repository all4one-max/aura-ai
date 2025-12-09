"""
Google Shopping API - SerpApi Integration

This module provides functionality to search Google Shopping using SerpApi.
"""

import os
from typing import List, Dict, Any, Optional

from serpapi import GoogleSearch

from app.schema import ChatQuery, Product
from app.tools.filtering import apply_filters

# Query string prefixes/suffixes (API-specific formatting)
QUERY_PREFIXES = {
    "size": "Size",
    "store": "from",
}

# TBS (To Be Searched) parameter mappings (SerpApi specific)
TBS_PARAMETERS = {
    "price": "price:1",
    "min_price": "ppr_min",
    "max_price": "ppr_max",
    "rating": "avg_rating",
    "sales": "sales:1",
    "shipping": "shipping:1",
    "new": "new:1",
    "used": "used:1",
    "sort_price_low": "sort:p",
    "sort_price_high": "sort:pd",
    "sort_rating_high": "sort:rv",
    "mr": "mr:1",
}

# Category pluralization map (for query string formatting)
CATEGORY_PLURAL_MAP = {
    "shoe": "shoes",
    "shoes": "shoes",
    "pant": "pants",
    "pants": "pants",
    "trouser": "trousers",
    "trousers": "trousers",
    "shirt": "shirt",
    "shirts": "shirt",
    "sunglass": "sunglasses",
    "sunglasses": "sunglasses",
    "dress": "dress",
    "dresses": "dress",
    "jacket": "jacket",
    "jackets": "jacket",
    "t-shirt": "t-shirt",
    "tshirt": "t-shirt",
    "top": "top",
    "tops": "top",
    "jean": "jeans",
    "jeans": "jeans",
}


def validate_product_fields(product: Dict[str, Any]) -> bool:
    """
    Validates that a product has all required fields based on Product schema.
    Uses API field mappings to check for alternative field names.
    """
    api_field_mapping = Product.get_api_field_mapping()
    required_fields = Product.get_required_fields()
    
    for field_name in required_fields:
        # Get possible API field names for this Product field
        api_field_names = api_field_mapping.get(field_name, [field_name])
        
        # Check if any of the API field names exist and have valid values
        field_found = False
        for api_field in api_field_names:
            value = product.get(api_field)
            if value is not None and (not isinstance(value, str) or value.strip() != ""):
                field_found = True
                break
        
        if not field_found:
            return False
    
    return True


def extract_product_data(product: Dict[str, Any]) -> Product:
    """
    Extracts and normalizes fields from a product using API field mappings from schema.
    Rating is optional and included if available.
    Returns a Product schema instance with validation.
    """
    api_field_mapping = Product.get_api_field_mapping()
    
    # Extract fields using API mappings (no hardcoded field names)
    image_fields = api_field_mapping.get("image", ["thumbnail", "image"])
    link_fields = api_field_mapping.get("link", ["link", "product_link"])
    price_fields = api_field_mapping.get("price", ["price"])
    
    # Get values with fallback to alternative field names
    image = next((product.get(f) for f in image_fields if product.get(f)), "")
    link = next((product.get(f) for f in link_fields if product.get(f)), "")
    price = next((product.get(f) for f in price_fields if product.get(f)), "")
    
    # Create Product instance (Pydantic will validate required fields)
    return Product(
        image=image,
        price=price,
        link=link,
        rating=product.get("rating", None),
        title=product.get("title", ""),
        source=product.get("source", ""),
        reviews=product.get("reviews", None),
    )


def chat_query_to_query_filters(chat_query: ChatQuery) -> Dict[str, Any]:
    """
    Converts ChatQuery to query_filters format for Google Shopping API.
    """
    query_filters = {
        "query": chat_query.query,
        "min_price": chat_query.min_price,
        "max_price": chat_query.max_price,
        "min_rating": chat_query.min_rating,
        "sort": chat_query.sort,
        "brand": chat_query.brand,
        "color": chat_query.color,
        "material": chat_query.material,
        "size": chat_query.size,
        "category": chat_query.category,
        "store": chat_query.store,
        "gender": chat_query.gender,
        "age_group": chat_query.age_group,
        "condition": chat_query.condition,
        "on_sale": chat_query.on_sale,
        "free_shipping": chat_query.free_shipping,
        "google_domain": chat_query.google_domain,
        "gl": chat_query.gl,
        "hl": chat_query.hl,
        "location": chat_query.location,
        "start": chat_query.start,
        "num": chat_query.num,
        "device": chat_query.device,
        "no_cache": chat_query.no_cache,
        "use_light_api": chat_query.use_light_api,
    }
    return query_filters


def search_google_shopping(query_filters: Dict[str, Any]) -> List[Product]:
    """
    Searches Google Shopping for the given query using SerpApi.
    Returns a list of products with validated required fields (image, price, link).
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("Error: SERPAPI_API_KEY environment variable not set.")
        print("Please get a key from https://serpapi.com/ and set it:")
        print("export SERPAPI_API_KEY='your_key_here'")
        return []

    # Strip whitespace from API key
    api_key = api_key.strip()

    # Extract query (required field)
    query = query_filters.get("query", "")
    if not query or (isinstance(query, str) and not query.strip()):
        print("Error: 'query' field is required in query_filters and cannot be empty.")
        return []

    # Extract filter parameters
    min_price = query_filters.get("min_price")
    max_price = query_filters.get("max_price")
    min_rating = query_filters.get("min_rating")
    sort = query_filters.get("sort")
    brand = query_filters.get("brand")
    color = query_filters.get("color")
    material = query_filters.get("material")
    size = query_filters.get("size")
    store = query_filters.get("store")
    gender = query_filters.get("gender")
    age_group = query_filters.get("age_group")
    category = query_filters.get("category")
    condition = query_filters.get("condition")
    on_sale = query_filters.get("on_sale", False)
    free_shipping = query_filters.get("free_shipping", False)

    # Extract additional parameters with defaults
    google_domain = query_filters.get("google_domain", "google.co.in")
    gl = query_filters.get("gl", "in")
    hl = query_filters.get("hl", "en")
    location = query_filters.get("location", "India")
    start = query_filters.get("start")
    num = query_filters.get("num")
    device = query_filters.get("device")
    no_cache = query_filters.get("no_cache", False)
    use_light_api = query_filters.get("use_light_api", False)

    # Handle filters that are best applied to the query string
    final_query = query.strip()

    # Prepend gender and age group for better results
    prefix_parts = []
    if gender:
        prefix_parts.append(gender.strip())
    if age_group:
        prefix_parts.append(age_group.strip())

    if prefix_parts:
        final_query = f"{' '.join(prefix_parts)} {final_query}"

    # Add category to query string (only if not already present)
    if category:
        category_clean = category.strip().lower()
        category_map = {
            "shoe": "shoes",
            "shoes": "shoes",
            "pant": "pants",
            "pants": "pants",
            "trouser": "trousers",
            "trousers": "trousers",
            "shirt": "shirt",
            "shirts": "shirt",
            "sunglass": "sunglasses",
            "sunglasses": "sunglasses",
            "dress": "dress",
            "dresses": "dress",
            "jacket": "jacket",
            "jackets": "jacket",
            "t-shirt": "t-shirt",
            "tshirt": "t-shirt",
            "top": "top",
            "tops": "top",
            "jean": "jeans",
            "jeans": "jeans",
        }
        category_term = category_map.get(category_clean, category_clean)
        # Check if category term is already in the query (case-insensitive)
        query_lower = final_query.lower()
        # Check for both singular and plural forms
        category_variants = [category_term, category_clean]
        if category_term.endswith('s') and not category_clean.endswith('s'):
            category_variants.append(category_term[:-1])  # singular form
        elif not category_term.endswith('s') and category_clean.endswith('s'):
            category_variants.append(category_term + 's')  # plural form
        
        # Only add if category term is not already in query
        # Filter out empty strings and check if any variant is in the query
        category_already_in_query = any(
            term and term in query_lower for term in category_variants
        )
        if not category_already_in_query:
            final_query += f" {category_term}"

    if brand:
        final_query += f" {brand.strip()}"
    if color:
        final_query += f" {color.strip()}"
    if material:
        final_query += f" {material.strip()}"
    if size:
        final_query += f" Size {size.strip()}"

    # Append store
    if store:
        final_query += f" from {store.strip()}"

    # Construct tbs (To Be Searched) parameter for filters
    tbs_parts = []

    # Price filter
    if min_price is not None or max_price is not None:
        price_filter = "price:1"
        if min_price is not None:
            price_filter += f",ppr_min:{int(min_price)}"
        if max_price is not None:
            price_filter += f",ppr_max:{int(max_price)}"
        tbs_parts.append(price_filter)

    # Rating filter
    if min_rating is not None:
        rating_val = int(min_rating * 100)
        tbs_parts.append(f"avg_rating:{rating_val}")

    # Sale filter
    if on_sale:
        tbs_parts.append("sales:1")

    # Shipping filter
    if free_shipping:
        tbs_parts.append("shipping:1")

    # Condition filter
    if condition:
        condition_lower = condition.lower().strip()
        if condition_lower == "new":
            tbs_parts.append("new:1")
        elif condition_lower == "used":
            tbs_parts.append("used:1")

    # Sort filter
    if sort:
        sort_lower = sort.lower().strip()
        if sort_lower == "price_low":
            tbs_parts.append("sort:p")
        elif sort_lower == "price_high":
            tbs_parts.append("sort:pd")
        elif sort_lower == "rating_high":
            tbs_parts.append("sort:rv")
        # relevance is default, no tbs needed

    # Combine filters
    if tbs_parts:
        tbs = "mr:1," + ",".join(tbs_parts)
    else:
        tbs = None

    # Determine which API to use
    engine = "google_shopping_light" if use_light_api else "google_shopping"

    params = {
        "engine": engine,
        "q": final_query,
        "api_key": api_key,
        "google_domain": google_domain,
        "gl": gl,
        "hl": hl,
        "location": location,
    }

    if tbs:
        params["tbs"] = tbs

    # Additional optional parameters
    if start is not None:
        params["start"] = start
    if num is not None:
        params["num"] = num
    if device:
        params["device"] = device
    if no_cache:
        params["no_cache"] = "true"

    print(f"Searching Google Shopping ({engine}) for: '{final_query}' in {location}...")

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        # Check for API errors
        if "error" in results:
            error_msg = results.get("error", "Unknown error")
            print(f"API Error: {error_msg}")
            if "invalid" in error_msg.lower() or "api key" in error_msg.lower():
                print("⚠️  Please verify your SERPAPI_API_KEY is correct.")
            return []

        # Check for shopping_results
        shopping_results = results.get("shopping_results", [])

        # Also check for organic_results as fallback
        if not shopping_results and "organic_results" in results:
            shopping_results = results.get("organic_results", [])

        if not shopping_results:
            if "products" in results:
                shopping_results = results.get("products", [])

            if not shopping_results:
                print("No products found in API response.")
                return []

        # Validate products with required fields
        validated_products = []
        skipped_count = 0

        for product in shopping_results:
            if not isinstance(product, dict):
                skipped_count += 1
                continue

            if validate_product_fields(product):
                product_data = extract_product_data(product)
                validated_products.append(product_data)
            else:
                skipped_count += 1

        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} product(s) due to missing required fields.")

        # Apply client-side filtering and sorting using the filtering module
        filtered_products, filter_stats = apply_filters(
            validated_products,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            sort=sort,
        )

        # Print filtering statistics
        if filter_stats["price_filtered"] > 0:
            print(f"⚠️ Filtered out {filter_stats['price_filtered']} product(s) outside price range.")
        
        if filter_stats["rating_filtered"] > 0:
            print(f"⚠️ Filtered out {filter_stats['rating_filtered']} product(s) below minimum rating.")

        print(f"✅ Found {len(filtered_products)} product(s) with all required fields.")
        return filtered_products

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        return []

