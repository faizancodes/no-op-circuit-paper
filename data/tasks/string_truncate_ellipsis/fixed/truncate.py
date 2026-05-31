def truncate(text: str, max_length: int, add_ellipsis: bool = False) -> str:
    """Truncate a string to a maximum length.

    Args:
        text: The string to truncate.
        max_length: Maximum length of the result.
        add_ellipsis: If True, append '...' to truncated strings.
                      Default is False (no ellipsis).

    Returns:
        The truncated string, optionally with ellipsis.
    """
    if len(text) <= max_length:
        return text
    
    if add_ellipsis:
        if max_length < 3:
            return text[:max_length]
        return text[:max_length - 3] + "..."
    else:
        return text[:max_length]
