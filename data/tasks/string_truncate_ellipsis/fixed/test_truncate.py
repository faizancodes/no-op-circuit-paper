from truncate import truncate

def test_truncate_without_ellipsis_by_default():
    assert truncate("Hello World", 5) == "Hello"

def test_truncate_with_explicit_ellipsis():
    assert truncate("Hello World", 5, add_ellipsis=True) == "He..."

def test_no_truncation_needed():
    assert truncate("Hi", 5) == "Hi"

def test_truncate_exact_length():
    assert truncate("Hello", 5) == "Hello"
