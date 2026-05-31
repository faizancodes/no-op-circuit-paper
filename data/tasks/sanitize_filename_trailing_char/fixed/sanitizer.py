def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from a filename.
    
    Removes characters that are not allowed in filenames on most systems:
    <, >, :, ", /, \, |, ?, *
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        A sanitized filename with invalid characters removed
    """
    invalid_chars = '<>:"/\\|?*'
    sanitized = ''.join(c for c in filename if c not in invalid_chars)
    return sanitized[:]
