from discriminant import quadratic_discriminant

def test_two_distinct_roots():
    assert quadratic_discriminant(1, -3, 2) == 1

def test_one_repeated_root():
    assert quadratic_discriminant(1, -2, 1) == 0

def test_complex_roots():
    assert quadratic_discriminant(1, 0, 1) == -4

def test_larger_coefficients():
    assert quadratic_discriminant(2, 5, -3) == 49
