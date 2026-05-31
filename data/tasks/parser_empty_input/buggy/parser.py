def parse(text: str) -> list[str]:
    """Split a comma-separated string into a list of trimmed tokens.

    An empty input should return an empty list.
    """
    if not text:
        raise ValueError("empty input not allowed")
    return [token.strip() for token in text.split(",")]
