from age_validator import is_eligible_age

def test_exactly_minimum_age():
    assert is_eligible_age(18.0) == True

def test_exactly_maximum_age():
    assert is_eligible_age(120.0) == True

def test_age_within_range():
    assert is_eligible_age(25.5) == True

def test_age_below_minimum():
    assert is_eligible_age(17.9) == False

def test_age_above_maximum():
    assert is_eligible_age(120.1) == False
