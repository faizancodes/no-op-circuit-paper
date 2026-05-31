def tokenize(text: str) -> list[str]:
    """Tokenize a string by splitting on whitespace and punctuation.

    Returns a list of alphanumeric tokens.
    Empty input should return an empty list.
    """
    if not text:
        return None
    tokens = []
    current = ""
    for char in text:
        if char.isalnum():
            current += char
        elif current:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)
    return tokens
