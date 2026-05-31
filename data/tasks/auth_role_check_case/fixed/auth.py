def check_user_role(user_role: str, required_role: str) -> bool:
    """Check if a user's role matches the required role.
    
    Role comparison should be case-insensitive.
    
    Args:
        user_role: The role assigned to the user
        required_role: The role required for access
        
    Returns:
        True if the user has the required role, False otherwise
    """
    return user_role.lower() == required_role.lower()
