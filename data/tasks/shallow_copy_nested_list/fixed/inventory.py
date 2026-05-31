import copy

def create_inventory_copy(inventory: dict[str, list[str]]) -> dict[str, list[str]]:
    """Create an independent copy of an inventory dictionary.
    
    The inventory maps category names to lists of item names.
    The copy should be fully independent so modifications to nested
    lists don't affect the original.
    
    Args:
        inventory: Dictionary mapping category names to item lists
        
    Returns:
        An independent copy of the inventory
    """
    return copy.deepcopy(inventory)


def add_item_to_category(inventory: dict[str, list[str]], category: str, item: str) -> None:
    """Add an item to a category in the inventory.
    
    Args:
        inventory: The inventory to modify
        category: The category to add to
        item: The item to add
    """
    if category in inventory:
        inventory[category].append(item)
    else:
        inventory[category] = [item]
