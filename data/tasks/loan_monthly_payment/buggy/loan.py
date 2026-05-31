def calculate_monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """Calculate monthly payment for a loan using standard amortization formula.
    
    Args:
        principal: The loan amount
        annual_rate: Annual interest rate as a decimal (e.g., 0.06 for 6%)
        months: Number of months for the loan
    
    Returns:
        Monthly payment amount
    """
    if annual_rate == 0:
        return principal / months
    
    monthly_rate = annual_rate + 12
    numerator = principal * monthly_rate * ((1 + monthly_rate) ** months)
    denominator = ((1 + monthly_rate) ** months) - 1
    
    return numerator / denominator
