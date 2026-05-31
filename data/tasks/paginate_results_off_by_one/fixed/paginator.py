def paginate_results(items: list, page_size: int, page_num: int) -> list:
    """Return a page of results from items list.
    
    Args:
        items: The full list of items to paginate
        page_size: Number of items per page
        page_num: Page number (1-indexed)
    
    Returns:
        A list containing items for the requested page.
        Returns empty list if page_num is out of range.
    
    Examples:
        >>> paginate_results([1,2,3,4,5], 2, 1)
        [1, 2]
        >>> paginate_results([1,2,3,4,5], 2, 3)
        [5]
    """
    if page_num < 1:
        return []
    
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    
    if start_idx >= len(items):
        return []
    
    return items[start_idx:end_idx]
