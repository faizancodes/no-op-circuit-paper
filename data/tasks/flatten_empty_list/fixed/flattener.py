def flatten(nested: list) -> list:
    """Flatten a nested list structure into a single-level list.
    
    Args:
        nested: A list that may contain nested lists at any depth.
    
    Returns:
        A flat list containing all elements in depth-first order.
        An empty input returns an empty list.
    
    Examples:
        >>> flatten([[1, 2], [3, 4]])
        [1, 2, 3, 4]
        >>> flatten([1, [2, [3, 4]]])
        [1, 2, 3, 4]
    """
    if not nested:
        return []
    
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
