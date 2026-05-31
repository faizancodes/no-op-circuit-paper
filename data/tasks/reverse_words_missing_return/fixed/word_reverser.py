def reverse_words(text: str) -> str:
    """Reverse each word in the string while preserving word order.
    
    Args:
        text: Input string with words separated by spaces
        
    Returns:
        String with each word reversed, maintaining original spacing
        
    Examples:
        >>> reverse_words("hello world")
        'olleh dlrow'
        >>> reverse_words("Python code")
        'nohtyP edoc'
    """
    if not text:
        return ""
    
    words = text.split(" ")
    reversed_words = [word[::-1] for word in words]
    result = " ".join(reversed_words)
    return result
