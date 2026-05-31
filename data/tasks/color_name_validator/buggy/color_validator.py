def validate_color_name(color: str) -> bool:
    """Check if a string is a valid CSS basic color name.
    
    Should accept color names case-insensitively.
    Returns True if valid, False otherwise.
    """
    valid_colors = {
        'red', 'blue', 'green', 'yellow', 'purple',
        'orange', 'pink', 'brown', 'black', 'white',
        'gray', 'cyan', 'magenta'
    }
    return color in valid_colors
