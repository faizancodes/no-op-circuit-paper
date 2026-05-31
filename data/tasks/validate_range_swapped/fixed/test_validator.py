from validator import validate_range

def test_value_in_middle_of_range():
    assert validate_range(5, 1, 10) == True

def test_value_at_lower_boundary():
    assert validate_range(1, 1, 10) == True

def test_value_at_upper_boundary():
    assert validate_range(10, 1, 10) == True

def test_value_below_range():
    assert validate_range(0, 1, 10) == False

def test_value_above_range():
    assert validate_range(11, 1, 10) == False
