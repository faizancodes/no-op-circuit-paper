from accumulator import accumulate_items

def test_single_call():
    result = accumulate_items('first')
    assert result == ['first']

def test_multiple_independent_calls():
    result1 = accumulate_items('a')
    result2 = accumulate_items('b')
    assert result2 == ['b'], f"Expected ['b'] but got {result2}"

def test_explicit_storage():
    my_list = []
    result1 = accumulate_items('x', my_list)
    result2 = accumulate_items('y', my_list)
    assert result2 == ['x', 'y']

def test_three_independent_calls():
    accumulate_items('first')
    accumulate_items('second')
    result = accumulate_items('third')
    assert result == ['third']
