def sort_products_by_price(products: list[dict]) -> list[dict]:
    """Sort products by price in descending order (most expensive first).
    
    Args:
        products: List of product dictionaries with 'name' and 'price' keys.
    
    Returns:
        List of products sorted by price, highest to lowest.
    """
    return sorted(products, key=lambda p: p['price'], reverse=True)
