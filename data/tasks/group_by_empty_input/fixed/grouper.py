from typing import Callable, TypeVar, Any
from collections import defaultdict

T = TypeVar('T')

def group_by(items: list[T], key: Callable[[T], Any]) -> dict[Any, list[T]]:
    """Group items by a key function.
    
    Returns a dictionary mapping each key to a list of items that
    produced that key. Empty input returns an empty dict.
    
    Args:
        items: List of items to group
        key: Function that extracts the grouping key from an item
        
    Returns:
        Dictionary mapping keys to lists of items
    """
    if not items:
        return {}
    
    groups = defaultdict(list)
    for item in items:
        k = key(item)
        groups[k].append(item)
    
    return dict(groups)
