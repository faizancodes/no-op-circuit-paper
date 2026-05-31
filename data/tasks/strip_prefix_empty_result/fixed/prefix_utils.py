def strip_prefix(text: str, prefix: str) -> str:
    """Remove the given prefix from text if it exists.
    
    If the prefix is not found at the start of text, return text unchanged.
    If the prefix matches the entire text, return an empty string.
    
    Args:
        text: The input string
        prefix: The prefix to remove
        
    Returns:
        The text with prefix removed, or original text if prefix not found
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
