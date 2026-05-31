from rotator import rotate_left

def test_empty_list():
    assert rotate_left([]) == []

def test_single_element():
    assert rotate_left([42]) == [42]

def test_two_elements():
    assert rotate_left([1, 2]) == [2, 1]

def test_multiple_elements():
    assert rotate_left([1, 2, 3, 4]) == [2, 3, 4, 1]

def test_strings():
    assert rotate_left(['a', 'b', 'c']) == ['b', 'c', 'a']
