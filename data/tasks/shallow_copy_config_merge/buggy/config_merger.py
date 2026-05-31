def merge_config(base: dict, overrides: dict) -> dict:
    """Merge override settings into a base configuration.
    
    Returns a new dictionary with base values updated by overrides.
    The returned config should be independent of the base.
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key].update(value)
        else:
            result[key] = value
    return result
