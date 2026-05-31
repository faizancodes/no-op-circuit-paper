def track_event(event: str, history=None) -> list[str]:
    """Track an event and return the current history.
    
    Args:
        event: The event name to track
        history: Optional list of previous events (default: empty list)
    
    Returns:
        List of all events in the current history
    """
    if history is None:
        history = []
    history.append(event)
    return history


def get_session_events(events: list[str]) -> list[str]:
    """Get a copy of the session events.
    
    Args:
        events: List of event names
    
    Returns:
        A copy of the event list
    """
    return events.copy()
