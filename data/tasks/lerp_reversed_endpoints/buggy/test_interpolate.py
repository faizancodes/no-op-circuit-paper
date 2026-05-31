from interpolate import lerp

def test_lerp_at_start():
    assert lerp(0, 10, 0.0) == 0.0

def test_lerp_at_end():
    assert lerp(0, 10, 1.0) == 10.0

def test_lerp_at_middle():
    assert lerp(0, 10, 0.5) == 5.0

def test_lerp_at_thirty_percent():
    assert lerp(0, 10, 0.3) == 3.0

def test_lerp_negative_range():
    assert lerp(10, -10, 0.25) == 5.0
