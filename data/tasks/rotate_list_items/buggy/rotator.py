def rotate_left(items: list) -> list:
    """Rotate a list left by one position.
    
    The first element moves to the end, and all other elements
    shift one position to the left.
    
    Examples:
        rotate_left([1, 2, 3]) -> [2, 3, 1]
        rotate_left([]) -> []
        rotate_left([5]) -> [5]
    """
    if len(items) <= 1:
        return items[:]
    return items[1:-1] + [items[0]]
