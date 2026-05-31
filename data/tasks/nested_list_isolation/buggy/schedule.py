def duplicate_schedule(schedule: list[list[str]]) -> list[list[str]]:
    """Create an independent copy of a weekly schedule.
    
    Each schedule is a list of days, where each day is a list of events.
    The returned copy should be fully independent of the original.
    
    Args:
        schedule: A list of lists representing a weekly schedule
        
    Returns:
        A new schedule that can be modified without affecting the original
    """
    return schedule.copy()


def add_event(schedule: list[list[str]], day_index: int, event: str) -> None:
    """Add an event to a specific day in the schedule.
    
    Args:
        schedule: The schedule to modify
        day_index: The index of the day (0-6)
        event: The event description to add
    """
    schedule[day_index].append(event)
