from statistics import median

def test_empty_list_returns_none():
    assert median([]) is None

def test_single_element():
    assert median([5.0]) == 5.0

def test_odd_number_of_elements():
    assert median([1.0, 3.0, 5.0]) == 3.0

def test_even_number_of_elements():
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5

def test_unsorted_input():
    assert median([5.0, 1.0, 3.0]) == 3.0
