import math
from sphere import sphere_volume

def test_sphere_volume_radius_one():
    result = sphere_volume(1)
    expected = (4 / 3) * math.pi
    assert abs(result - expected) < 1e-9

def test_sphere_volume_radius_three():
    result = sphere_volume(3)
    expected = (4 / 3) * math.pi * 27
    assert abs(result - expected) < 1e-9

def test_sphere_volume_radius_zero():
    assert sphere_volume(0) == 0.0

def test_sphere_volume_radius_five():
    result = sphere_volume(5)
    expected = (4 / 3) * math.pi * 125
    assert abs(result - expected) < 1e-9
