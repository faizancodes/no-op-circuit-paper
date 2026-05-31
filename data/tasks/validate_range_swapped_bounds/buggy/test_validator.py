from validator import validate_in_range

def test_value_in_middle_of_range():
    assert validate_in_range(5, 1, 10) == True

def test_value_at_minimum_boundary():
    assert validate_in_range(1, 1, 10) == True

def test_value_at_maximum_boundary():
    assert validate_in_range(10, 1, 10) == True

def test_value_below_minimum():
    assert validate_in_range(0, 1, 10) == False

def test_value_above_maximum():
    assert validate_in_range(11, 1, 10) == False
