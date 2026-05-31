from validator import is_valid_password

def test_exactly_eight_characters_with_digit():
    assert is_valid_password("abcd1234") == True

def test_nine_characters_with_digit():
    assert is_valid_password("abcde1234") == True

def test_too_short_with_digit():
    assert is_valid_password("abc123") == False

def test_long_enough_but_no_digit():
    assert is_valid_password("abcdefgh") == False
