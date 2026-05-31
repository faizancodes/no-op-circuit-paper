from lookup import find_item

def test_case_insensitive_default():
    inventory = [{'name': 'apple', 'quantity': 10}, {'name': 'banana', 'quantity': 5}]
    result = find_item(inventory, 'APPLE')
    assert result == {'name': 'apple', 'quantity': 10}

def test_explicit_case_sensitive():
    inventory = [{'name': 'apple', 'quantity': 10}, {'name': 'banana', 'quantity': 5}]
    result = find_item(inventory, 'APPLE', case_sensitive=True)
    assert result is None

def test_exact_match():
    inventory = [{'name': 'apple', 'quantity': 10}, {'name': 'banana', 'quantity': 5}]
    result = find_item(inventory, 'apple')
    assert result == {'name': 'apple', 'quantity': 10}

def test_not_found():
    inventory = [{'name': 'apple', 'quantity': 10}]
    result = find_item(inventory, 'orange')
    assert result is None
