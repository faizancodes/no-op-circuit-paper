import math

def sphere_volume(radius: float) -> float:
    """Calculate the volume of a sphere with given radius.
    
    Formula: V = (4/3) * π * r³
    
    Args:
        radius: The radius of the sphere (must be non-negative)
    
    Returns:
        The volume of the sphere
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return (4 / 3) * math.pi * (radius ** 3)
