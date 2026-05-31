from rotation import rotate_left

def test_rotate_by_zero():
    assert rotate_left([1, 2, 3], 0) == [1, 2, 3]

def test_rotate_by_one():
    assert rotate_left([1, 2, 3, 4], 1) == [2, 3, 4, 1]

def test_rotate_by_length():
    assert rotate_left([1, 2, 3], 3) == [1, 2, 3]

def test_rotate_empty_list():
    assert rotate_left([], 5) == []

def test_rotate_by_more_than_length():
    assert rotate_left([1, 2, 3], 5) == [3, 1, 2]
