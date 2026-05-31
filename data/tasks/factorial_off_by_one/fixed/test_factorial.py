from factorial import factorial

def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_one():
    assert factorial(1) == 1

def test_factorial_small():
    assert factorial(3) == 6
    assert factorial(4) == 24

def test_factorial_five():
    assert factorial(5) == 120

def test_factorial_large():
    assert factorial(10) == 3628800
