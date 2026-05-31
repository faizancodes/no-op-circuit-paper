from chunks import partition_into_chunks

def test_even_division():
    assert partition_into_chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

def test_uneven_division_with_remainder():
    assert partition_into_chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

def test_single_element_chunks():
    assert partition_into_chunks([1, 2, 3], 1) == [[1], [2], [3]]

def test_chunk_size_larger_than_list():
    assert partition_into_chunks([1, 2], 5) == [[1, 2]]

def test_empty_list():
    assert partition_into_chunks([], 3) == []
