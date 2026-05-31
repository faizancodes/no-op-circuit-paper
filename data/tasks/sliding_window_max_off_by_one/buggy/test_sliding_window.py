from sliding_window import sliding_window_max

def test_basic_window():
    assert sliding_window_max([1, 3, 2, 5, 4], 3) == [3, 5, 5]

def test_window_size_one():
    assert sliding_window_max([1, 2, 3, 4], 1) == [1, 2, 3, 4]

def test_window_equals_list_size():
    assert sliding_window_max([5, 2, 8, 1], 4) == [8]

def test_two_element_window():
    assert sliding_window_max([1, 2, 3, 4, 5], 2) == [2, 3, 4, 5]

def test_empty_list():
    assert sliding_window_max([], 1) == []
