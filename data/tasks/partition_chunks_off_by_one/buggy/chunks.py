def partition_into_chunks(items: list, chunk_size: int) -> list[list]:
    """Partition a list into chunks of the given size.

    The last chunk may be smaller if the list length is not evenly divisible.

    Args:
        items: The list to partition
        chunk_size: Size of each chunk (must be positive)

    Returns:
        A list of chunks, where each chunk is a list

    Example:
        >>> partition_into_chunks([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    result = []
    n = len(items)
    num_chunks = n // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        result.append(items[start:end])
    
    return result
