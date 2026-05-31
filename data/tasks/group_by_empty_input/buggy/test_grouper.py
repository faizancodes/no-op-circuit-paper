from grouper import group_by

def test_empty_list_returns_empty_dict():
    result = group_by([], key=lambda x: x)
    assert result == {}
    assert isinstance(result, dict)

def test_group_by_simple_key():
    items = [1, 2, 3, 4, 5, 6]
    result = group_by(items, key=lambda x: x % 2)
    assert result == {0: [2, 4, 6], 1: [1, 3, 5]}

def test_group_by_string_length():
    words = ["a", "bb", "ccc", "dd", "e"]
    result = group_by(words, key=len)
    assert result == {1: ["a", "e"], 2: ["bb", "dd"], 3: ["ccc"]}

def test_group_by_first_letter():
    words = ["apple", "ant", "banana", "berry", "cherry"]
    result = group_by(words, key=lambda w: w[0])
    assert result == {"a": ["apple", "ant"], "b": ["banana", "berry"], "c": ["cherry"]}
