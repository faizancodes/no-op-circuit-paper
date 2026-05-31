def collect_tags(tag: str, collection=None) -> list[str]:
    """Collect a tag into a collection and return the collection.
    
    Args:
        tag: The tag to add to the collection
        collection: Optional collection to add to (defaults to empty list)
    
    Returns:
        List containing the tag(s)
    """
    if collection is None:
        collection = []
    collection.append(tag)
    return collection
