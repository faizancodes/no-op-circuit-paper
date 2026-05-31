from parser import parse


def test_empty_input_returns_empty_list():
    assert parse("") == []


def test_basic_split():
    assert parse("a,b,c") == ["a", "b", "c"]


def test_whitespace_trimmed():
    assert parse("a , b ,c") == ["a", "b", "c"]
