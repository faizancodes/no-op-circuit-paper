def match_substring(text: str, pattern: str, case_sensitive: bool = True) -> bool:
    """Check if pattern exists as a substring in text.

    Args:
        text: The text to search in.
        pattern: The substring pattern to find.
        case_sensitive: Whether matching should be case-sensitive.
                       Defaults to False (case-insensitive).

    Returns:
        True if pattern is found in text, False otherwise.
    """
    if not case_sensitive:
        text = text.lower()
        pattern = pattern.lower()
    return pattern in text
