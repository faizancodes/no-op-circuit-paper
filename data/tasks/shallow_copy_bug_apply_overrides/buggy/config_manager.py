def apply_overrides(base_config: dict, overrides: dict) -> dict:
    """Apply overrides to a base configuration.
    
    Returns a new configuration dict with overrides applied.
    The original base_config should not be modified.
    
    Args:
        base_config: The base configuration dictionary
        overrides: Dictionary of override values to apply
    
    Returns:
        A new configuration dictionary with overrides applied
    """
    result = base_config.copy()
    for key, value in overrides.items():
        result[key] = value
    return result
