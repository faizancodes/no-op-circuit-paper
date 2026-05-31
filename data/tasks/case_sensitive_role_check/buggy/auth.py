def check_permission(user_role: str, required_role: str) -> bool:
    """Check if a user role has permission for a required role.
    
    Roles are hierarchical: admin > moderator > user.
    Role comparison should be case-insensitive.
    
    Args:
        user_role: The role of the user
        required_role: The minimum role required
    
    Returns:
        True if user_role has sufficient permissions, False otherwise
    """
    role_hierarchy = {
        "admin": 3,
        "moderator": 2,
        "user": 1
    }
    
    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role.lower(), 0)
    
    return user_level >= required_level
