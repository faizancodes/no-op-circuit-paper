from prefix_utils import strip_prefix

def test_prefix_equals_entire_string():
    assert strip_prefix("hello", "hello") == ""

def test_basic_prefix_removal():
    assert strip_prefix("hello world", "hello ") == "world"

def test_no_prefix_match():
    assert strip_prefix("hello", "world") == "hello"

def test_empty_prefix():
    assert strip_prefix("hello", "") == "hello"
