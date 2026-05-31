def sliding_window_max(nums: list[int], k: int) -> list[int]:
    """Return the maximum value in each sliding window of size k.
    
    For a list of length n, there are (n - k + 1) valid windows.
    Each window starts at position i where 0 <= i <= n - k.
    
    Args:
        nums: List of integers
        k: Window size (must be > 0 and <= len(nums))
    
    Returns:
        List of maximum values, one per window position
    
    Example:
        >>> sliding_window_max([1, 3, 2, 5, 4], 3)
        [3, 5, 5]
    """
    if not nums or k <= 0 or k > len(nums):
        return []
    
    result = []
    n = len(nums)
    
    for i in range(n - k):
        window = nums[i:i + k]
        result.append(max(window))
    
    return result
