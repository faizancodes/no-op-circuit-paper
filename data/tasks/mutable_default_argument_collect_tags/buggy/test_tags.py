from tags import collect_tags

def test_single_tag():
    result = collect_tags('python')
    assert result == ['python']

def test_independent_calls():
    first = collect_tags('python')
    second = collect_tags('rust')
    assert second == ['rust'], f"Expected ['rust'], got {second}"

def test_with_explicit_collection():
    existing = ['java']
    result = collect_tags('kotlin', existing)
    assert result == ['java', 'kotlin']

def test_multiple_independent_calls():
    result1 = collect_tags('go')
    result2 = collect_tags('swift')
    result3 = collect_tags('ruby')
    assert result1 == ['go']
    assert result2 == ['swift']
    assert result3 == ['ruby']
