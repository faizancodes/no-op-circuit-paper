from tokenizer import tokenize

def test_empty_input_returns_empty_list():
    assert tokenize("") == []

def test_basic_tokenization():
    assert tokenize("hello world") == ["hello", "world"]

def test_punctuation_separated():
    assert tokenize("hello,world!test") == ["hello", "world", "test"]

def test_mixed_separators():
    assert tokenize("foo-bar_baz") == ["foo", "bar", "baz"]
