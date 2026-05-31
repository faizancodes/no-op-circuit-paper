from matcher import match_substring

def test_default_case_insensitive():
    assert match_substring('Hello World', 'hello') == True

def test_explicit_case_sensitive():
    assert match_substring('Hello World', 'hello', case_sensitive=True) == False

def test_explicit_case_insensitive():
    assert match_substring('Hello World', 'WORLD', case_sensitive=False) == True

def test_exact_match_default():
    assert match_substring('test', 'test') == True
