def validate_range(value: float, minimum: float, maximum: float) -> bool:
    """Check if value is within the specified range [minimum, maximum].

    Args:
        value: The value to check
        minimum: The minimum allowed value (inclusive)
        maximum: The maximum allowed value (inclusive)

    Returns:
        True if minimum <= value <= maximum, False otherwise
    """
    return _is_in_range(value, minimum, maximum)


def _is_in_range(val: float, low: float, high: float) -> bool:
    """Helper to check if val is in [low, high]."""
    return low <= val <= high
