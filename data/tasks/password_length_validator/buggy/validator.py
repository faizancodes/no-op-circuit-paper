def is_valid_password(password: str) -> bool:
    """Check if a password meets the minimum requirements.
    
    A valid password must:
    - Be at least 8 characters long
    - Contain at least one digit
    
    Args:
        password: The password string to validate
        
    Returns:
        True if the password is valid, False otherwise
    """
    if len(password) > 8 and any(c.isdigit() for c in password):
        return True
    return False
