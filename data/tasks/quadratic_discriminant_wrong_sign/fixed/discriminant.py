def quadratic_discriminant(a: float, b: float, c: float) -> float:
    """Calculate the discriminant of a quadratic equation ax² + bx + c = 0.
    
    The discriminant is b² - 4ac and determines the nature of the roots:
    - Positive: two distinct real roots
    - Zero: one repeated real root
    - Negative: two complex conjugate roots
    
    Args:
        a: coefficient of x²
        b: coefficient of x
        c: constant term
    
    Returns:
        The discriminant value
    """
    return b * b - 4 * a * c
