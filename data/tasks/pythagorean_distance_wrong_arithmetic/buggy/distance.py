import math

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the Euclidean distance between two 2D points.
    
    Args:
        x1, y1: Coordinates of the first point
        x2, y2: Coordinates of the second point
    
    Returns:
        The Euclidean distance between the two points
    """
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx**2 + dy**2 + dx + dy)
