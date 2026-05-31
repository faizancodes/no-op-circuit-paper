def collect_tags(tag: str, collection=[]) -> list[str]:
    """Collect a tag into a collection and return the collection.
    
    Args:
        tag: The tag to add to the collection
        collection: Optional collection to add to (defaults to empty list)
    
    Returns:
        List containing the tag(s)
    """
    collection.append(tag)
    return collection
