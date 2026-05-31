from paginator import paginate_results

def test_first_page_full():
    items = list(range(10))
    assert paginate_results(items, 3, 1) == [0, 1, 2]

def test_middle_page_full():
    items = list(range(10))
    assert paginate_results(items, 3, 2) == [3, 4, 5]

def test_last_page_partial():
    items = list(range(10))
    assert paginate_results(items, 3, 4) == [9]

def test_single_item_last_page():
    items = [1, 2, 3, 4, 5]
    assert paginate_results(items, 2, 3) == [5]

def test_out_of_range_page():
    items = list(range(5))
    assert paginate_results(items, 3, 5) == []
