def is_within_bounds(value: float, lower: float, upper: float) -> bool:
    """Check if value is between lower and upper (inclusive)."""
    return lower <= value <= upper


def validate_in_range(value: float, minimum: float, maximum: float) -> bool:
    """Validate that a value falls within the specified range.
    
    Args:
        value: The value to validate
        minimum: The minimum allowed value (inclusive)
        maximum: The maximum allowed value (inclusive)
    
    Returns:
        True if value is in [minimum, maximum], False otherwise
    """
    return is_within_bounds(value, minimum, maximum)
