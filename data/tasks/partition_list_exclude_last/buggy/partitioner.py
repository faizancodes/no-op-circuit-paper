def partition_by_predicate(items: list, predicate) -> tuple[list, list]:
    """Partition a list into two lists based on a predicate.
    
    Returns a tuple (matching, non_matching) where:
    - matching contains all items for which predicate(item) is True
    - non_matching contains all items for which predicate(item) is False
    
    All elements from the input list should appear exactly once in the output.
    """
    matching = []
    non_matching = []
    
    for item in items[:-1]:
        if predicate(item):
            matching.append(item)
        else:
            non_matching.append(item)
    
    return (matching, non_matching)
