from partitioner import partition_by_predicate

def test_partition_even_odd():
    matching, non_matching = partition_by_predicate([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
    assert matching == [2, 4]
    assert non_matching == [1, 3, 5]

def test_partition_empty_list():
    matching, non_matching = partition_by_predicate([], lambda x: x > 0)
    assert matching == []
    assert non_matching == []

def test_partition_all_matching():
    matching, non_matching = partition_by_predicate([2, 4, 6, 8], lambda x: x % 2 == 0)
    assert matching == [2, 4, 6, 8]
    assert non_matching == []

def test_partition_strings():
    matching, non_matching = partition_by_predicate(["apple", "banana", "avocado", "cherry"], lambda s: s.startswith("a"))
    assert matching == ["apple", "avocado"]
    assert non_matching == ["banana", "cherry"]
