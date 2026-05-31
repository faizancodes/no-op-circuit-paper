from rotator import rotate_list

def test_rotate_right_by_one():
    assert rotate_list([1, 2, 3, 4], 1) == [4, 1, 2, 3]

def test_rotate_left_by_one():
    assert rotate_list([1, 2, 3, 4], -1) == [2, 3, 4, 1]

def test_rotate_by_length():
    assert rotate_list([1, 2, 3], 3) == [1, 2, 3]

def test_empty_list():
    assert rotate_list([], 5) == []

def test_rotate_by_zero():
    assert rotate_list([1, 2, 3], 0) == [1, 2, 3]
