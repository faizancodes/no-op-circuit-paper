from distance import euclidean_distance
import math

def test_standard_3_4_5_triangle():
    assert euclidean_distance(0, 0, 3, 4) == 5.0

def test_same_point_zero_distance():
    assert euclidean_distance(5, 5, 5, 5) == 0.0

def test_unit_distance_horizontal():
    assert euclidean_distance(0, 0, 1, 0) == 1.0

def test_diagonal_distance():
    result = euclidean_distance(0, 0, 1, 1)
    assert abs(result - math.sqrt(2)) < 1e-10

def test_negative_coordinates():
    assert euclidean_distance(-3, -4, 0, 0) == 5.0
