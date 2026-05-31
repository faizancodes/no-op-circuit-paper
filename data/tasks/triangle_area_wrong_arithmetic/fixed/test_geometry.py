from geometry import triangle_area

def test_triangle_area_basic():
    assert triangle_area(10, 5) == 25.0

def test_triangle_area_different_values():
    assert triangle_area(6, 4) == 12.0

def test_triangle_area_unit_triangle():
    assert triangle_area(2, 2) == 2.0

def test_triangle_area_large_values():
    assert triangle_area(100, 50) == 2500.0
