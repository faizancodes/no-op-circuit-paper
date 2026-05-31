def register_action(action: str, log: list = None) -> list:
    """Register an action in a log and return the updated log.
    
    Args:
        action: The action to register.
        log: The log to append to. Defaults to an empty list.
    
    Returns:
        The log with the new action appended.
    """
    if log is None:
        log = []
    log.append(action)
    return log
