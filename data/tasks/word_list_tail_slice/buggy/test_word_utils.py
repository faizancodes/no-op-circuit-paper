from word_utils import get_all_but_first_word

def test_empty_string():
    assert get_all_but_first_word("") == []

def test_single_word():
    assert get_all_but_first_word("hello") == []

def test_two_words():
    assert get_all_but_first_word("hello world") == ["world"]

def test_three_words():
    assert get_all_but_first_word("hello world test") == ["world", "test"]

def test_multiple_words():
    assert get_all_but_first_word("the quick brown fox jumps") == ["quick", "brown", "fox", "jumps"]
