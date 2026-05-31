def transpose_matrix(matrix: list[list[int]]) -> list[list[int]]:
    """Transpose a 2D matrix (swap rows and columns).
    
    Args:
        matrix: A 2D list representing the matrix
        
    Returns:
        The transposed matrix
        
    Examples:
        >>> transpose_matrix([[1, 2], [3, 4]])
        [[1, 3], [2, 4]]
    """
    if not matrix or not matrix[0]:
        return []
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    result = [[matrix[row][col] for row in range(rows)] for col in range(cols)]
