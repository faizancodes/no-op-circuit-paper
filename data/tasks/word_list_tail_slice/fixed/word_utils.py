def get_all_but_first_word(sentence: str) -> list[str]:
    """Return all words in the sentence except the first one.
    
    Args:
        sentence: A string containing space-separated words.
    
    Returns:
        A list of all words except the first.
        Empty list if sentence has zero or one word.
    
    Examples:
        >>> get_all_but_first_word("hello world")
        ["world"]
        >>> get_all_but_first_word("hello")
        []
    """
    words = sentence.split()
    if len(words) <= 1:
        return []
    return words[1:]
