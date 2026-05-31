def accumulate_items(item, storage=[]):
    """Add an item to a storage list and return it.
    
    Args:
        item: The item to add to storage
        storage: Optional list to accumulate items into. If not provided,
                 a fresh empty list is used.
    
    Returns:
        The storage list with the new item appended.
    """
    storage.append(item)
    return storage
