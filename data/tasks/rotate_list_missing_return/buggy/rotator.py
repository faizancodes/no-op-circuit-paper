def rotate_list(items: list, positions: int) -> list:
    """Rotate a list to the right by the given number of positions.

    Args:
        items: The list to rotate
        positions: Number of positions to rotate right (negative for left)

    Returns:
        A new list rotated by the specified positions

    Examples:
        >>> rotate_list([1, 2, 3, 4], 1)
        [4, 1, 2, 3]
        >>> rotate_list([1, 2, 3, 4], -1)
        [2, 3, 4, 1]
    """
    if not items:
        return []
    
    n = len(items)
    positions = positions % n
    
    rotated = items[-positions:] + items[:-positions]
