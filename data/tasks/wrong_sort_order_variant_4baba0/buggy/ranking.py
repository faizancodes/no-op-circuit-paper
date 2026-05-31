def rank_students(students: list[dict]) -> list[dict]:
    """Rank students by their score in descending order (highest first).
    
    Each student is a dict with keys 'name' and 'score'.
    Returns a new list sorted by score, highest to lowest.
    """
    return sorted(students, key=lambda s: s['score'])
