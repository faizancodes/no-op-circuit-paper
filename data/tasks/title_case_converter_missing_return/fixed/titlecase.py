def convert_to_title_case(text: str) -> str:
    """Convert a string to title case.
    
    Each word's first letter is capitalized, and the rest are lowercase.
    Words are separated by spaces.
    
    Args:
        text: The input string to convert
        
    Returns:
        The title-cased string
    """
    if not text:
        return ""
    
    words = text.split()
    result = []
    
    for word in words:
        if word:
            titled = word[0].upper() + word[1:].lower()
            result.append(titled)
    
    return " ".join(result)
