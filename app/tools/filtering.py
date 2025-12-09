"""
Client-side filtering for product search results.

This module handles filtering and sorting of products that the API
may not strictly enforce, ensuring consistent results.
"""

from typing import List, Optional

from app.schema import Product


def _extract_price(price_str: str, currency_symbols: Optional[List[str]] = None) -> float:
    """
    Helper function to extract numeric price from price string.
    Returns 0.0 if parsing fails.
    
    Args:
        price_str: Price string (e.g., "₹1,999", "$2,499.50", "€100")
        currency_symbols: Optional list of currency symbols to remove.
                        If None, uses common currency symbols.
        
    Returns:
        Numeric price as float, or 0.0 if parsing fails
    """
    if currency_symbols is None:
        # Common currency symbols - can be extended
        currency_symbols = ['₹', '$', '€', '£', '¥', 'Rs', 'rs']
    
    try:
        price_clean = str(price_str)
        
        # Remove currency symbols
        for symbol in currency_symbols:
            price_clean = price_clean.replace(symbol, '')
        
        # Remove thousands separators and spaces
        price_clean = price_clean.replace(',', '').replace(' ', '').strip()
        
        return float(price_clean)
    except (ValueError, AttributeError):
        return 0.0


def filter_by_price(
    products: List[Product],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> tuple[List[Product], int]:
    """
    Filter products by price range.
    
    Args:
        products: List of products to filter
        min_price: Minimum price (inclusive)
        max_price: Maximum price (inclusive)
        
    Returns:
        Tuple of (filtered_products, filtered_count)
    """
    if min_price is None and max_price is None:
        return products, 0
    
    filtered_products = []
    filtered_count = 0
    
    for product in products:
        try:
            price_num = _extract_price(product.price)
            
            # Check if price is within range
            if min_price is not None and price_num < min_price:
                filtered_count += 1
                continue
            if max_price is not None and price_num > max_price:
                filtered_count += 1
                continue
            
            filtered_products.append(product)
        except (ValueError, AttributeError):
            # If price parsing fails, include the product
            filtered_products.append(product)
    
    return filtered_products, filtered_count


def filter_by_rating(
    products: List[Product],
    min_rating: Optional[float] = None,
) -> tuple[List[Product], int]:
    """
    Filter products by minimum rating.
    
    Args:
        products: List of products to filter
        min_rating: Minimum rating (inclusive, 0-5 scale)
        
    Returns:
        Tuple of (filtered_products, filtered_count)
    """
    if min_rating is None:
        return products, 0
    
    filtered_products = []
    filtered_count = 0
    
    for product in products:
        if product.rating is None or product.rating < min_rating:
            filtered_count += 1
            continue
        
        filtered_products.append(product)
    
    return filtered_products, filtered_count


def sort_products(
    products: List[Product],
    sort: Optional[str] = None,
) -> List[Product]:
    """
    Sort products based on sort criteria.
    
    Args:
        products: List of products to sort
        sort: Sort order - "price_low", "price_high", "rating_high", or None
        
    Returns:
        Sorted list of products
    """
    if not sort or not products:
        return products
    
    sort_lower = sort.lower().strip()
    
    if sort_lower == "price_low":
        return sorted(products, key=lambda p: _extract_price(p.price))
    elif sort_lower == "price_high":
        return sorted(products, key=lambda p: _extract_price(p.price), reverse=True)
    elif sort_lower == "rating_high":
        return sorted(products, key=lambda p: p.rating if p.rating else 0, reverse=True)
    else:
        # Unknown sort option, return as-is
        return products


def apply_filters(
    products: List[Product],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    sort: Optional[str] = None,
) -> tuple[List[Product], dict[str, int]]:
    """
    Apply all filters and sorting to products.
    
    Args:
        products: List of products to filter and sort
        min_price: Minimum price filter
        max_price: Maximum price filter
        min_rating: Minimum rating filter
        sort: Sort order
        
    Returns:
        Tuple of (filtered_and_sorted_products, filter_stats)
        filter_stats contains counts of filtered products
    """
    filter_stats = {
        "price_filtered": 0,
        "rating_filtered": 0,
    }
    
    # Apply price filter
    products, price_count = filter_by_price(products, min_price, max_price)
    filter_stats["price_filtered"] = price_count
    
    # Apply rating filter
    products, rating_count = filter_by_rating(products, min_rating)
    filter_stats["rating_filtered"] = rating_count
    
    # Apply sorting
    products = sort_products(products, sort)
    
    return products, filter_stats

