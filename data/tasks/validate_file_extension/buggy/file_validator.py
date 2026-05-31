def validate_file_extension(filename: str) -> bool:
    """Check if a filename has a valid extension.

    Valid extensions (case-insensitive): .pdf, .txt, .jpg, .png, .doc
    
    Args:
        filename: The name of the file to validate
    
    Returns:
        True if the file has a valid extension, False otherwise
    """
    valid_extensions = [".pdf", ".txt", ".jpg", ".png", ".doc"]
    
    if not filename or "." not in filename:
        return False
    
    # Extract the extension
    extension = filename[filename.rfind("."):]
    
    return extension in valid_extensions
