def sort_books_by_year(books: list[dict]) -> list[dict]:
    """Sort a list of books by publication year in descending order.
    
    Each book is a dictionary with at least a 'year' key.
    Returns a new list with books sorted from newest to oldest.
    
    Args:
        books: List of book dictionaries with 'year' keys
        
    Returns:
        New list sorted by year, descending (newest first)
    """
    return sorted(books, key=lambda b: b['year'], reverse=True)
