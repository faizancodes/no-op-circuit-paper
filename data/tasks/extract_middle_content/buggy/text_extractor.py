def extract_middle(text: str, delimiter: str = '|') -> str:
    """Extract content from first to last delimiter (inclusive).
    
    Returns the substring starting from the first occurrence of the delimiter
    up to and including the last occurrence of the delimiter.
    If delimiter appears only once, returns the delimiter itself.
    If delimiter not found, returns empty string.
    
    Args:
        text: Input string to search
        delimiter: Character to use as boundary marker
        
    Returns:
        Extracted content including both delimiters, or empty string if not found
    """
    first_idx = text.find(delimiter)
    if first_idx == -1:
        return ""
    
    last_idx = text.rfind(delimiter)
    return text[first_idx:last_idx]
