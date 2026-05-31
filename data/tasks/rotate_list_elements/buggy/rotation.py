def rotate_left(items: list, n: int) -> list:
    """Rotate a list to the left by n positions.
    
    Args:
        items: The list to rotate
        n: Number of positions to rotate left
        
    Returns:
        A new list with elements rotated left by n positions.
        Rotation wraps around, so rotating by len(items) returns the original.
    """
    if not items:
        return []
    
    n = n % len(items)
    result = []
    
    for i in range(len(items)):
        result.append(items[(i + n) % len(items)])
    
    return result[:len(items) - n] if n > 0 else result
