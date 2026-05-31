from flattener import flatten

def test_empty_list_returns_empty_list():
    assert flatten([]) == []

def test_flat_list_unchanged():
    assert flatten([1, 2, 3]) == [1, 2, 3]

def test_nested_two_levels():
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

def test_deeply_nested():
    assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]

def test_mixed_nesting():
    assert flatten([1, [2, 3], 4, [5, [6, 7]]]) == [1, 2, 3, 4, 5, 6, 7]
