def is_eligible_age(age: float, min_age: float = 18.0, max_age: float = 120.0) -> bool:
    """Check if an age is within the eligible range.
    
    Args:
        age: The age to check.
        min_age: Minimum eligible age (inclusive, default 18.0).
        max_age: Maximum eligible age (inclusive, default 120.0).
    
    Returns:
        True if age is within [min_age, max_age], False otherwise.
    """
    if age < 0:
        return False
    if age >= min_age and age <= max_age:
        return True
    return False
