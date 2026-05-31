def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation between start and end.
    
    Args:
        start: The starting value (at t=0)
        end: The ending value (at t=1)
        t: The interpolation parameter (0.0 to 1.0)
    
    Returns:
        The interpolated value
    """
    return _interpolate(start, end, t)


def _interpolate(a: float, b: float, factor: float) -> float:
    """Helper function to compute interpolation.
    
    Args:
        a: First value
        b: Second value
        factor: Blend factor
    
    Returns:
        a * (1 - factor) + b * factor
    """
    return a * (1 - factor) + b * factor
