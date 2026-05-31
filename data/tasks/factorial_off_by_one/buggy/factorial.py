def factorial(n: int) -> int:
    """Compute the factorial of n.
    
    Args:
        n: A non-negative integer
    
    Returns:
        The factorial of n (n!)
    
    Examples:
        factorial(0) = 1
        factorial(5) = 120
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    result = 1
    for i in range(1, n):
        result *= i
    return result
