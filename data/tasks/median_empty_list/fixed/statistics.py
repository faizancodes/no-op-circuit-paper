def median(values: list[float]) -> float | None:
    """Calculate the median of a list of numbers.

    Returns None if the input list is empty.
    """
    if not values:
        return None
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    else:
        return sorted_values[mid]
