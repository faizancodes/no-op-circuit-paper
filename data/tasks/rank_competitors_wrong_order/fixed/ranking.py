def rank_competitors(competitors: list[dict]) -> list[dict]:
    """Rank competitors by score in descending order.

    Args:
        competitors: List of dicts with 'name' and 'score' keys.

    Returns:
        List of competitors sorted from highest to lowest score.
    """
    return sorted(competitors, key=lambda c: c['score'], reverse=True)
