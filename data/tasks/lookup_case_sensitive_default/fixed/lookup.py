def find_item(inventory: list[dict], name: str, case_sensitive: bool = False) -> dict | None:
    """Find an item in the inventory by name.
    
    Args:
        inventory: List of item dictionaries with 'name' and 'quantity' keys.
        name: The item name to search for.
        case_sensitive: Whether to perform case-sensitive lookup. Default is False
                       for case-insensitive matching.
    
    Returns:
        The matching item dictionary, or None if not found.
    """
    for item in inventory:
        item_name = item.get('name', '')
        if case_sensitive:
            if item_name == name:
                return item
        else:
            if item_name.lower() == name.lower():
                return item
    return None
