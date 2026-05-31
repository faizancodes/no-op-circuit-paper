from schedule import duplicate_schedule, add_event

def test_duplicate_creates_separate_schedule():
    original = [["meeting"], ["lunch"], ["gym"]]
    copy = duplicate_schedule(original)
    add_event(copy, 0, "dinner")
    assert original[0] == ["meeting"]
    assert copy[0] == ["meeting", "dinner"]

def test_duplicate_preserves_content():
    original = [["task1", "task2"], ["task3"]]
    copy = duplicate_schedule(original)
    assert copy == [["task1", "task2"], ["task3"]]

def test_empty_schedule_duplication():
    original = [[], [], []]
    copy = duplicate_schedule(original)
    add_event(copy, 1, "new_event")
    assert original[1] == []
    assert copy[1] == ["new_event"]

def test_multiple_modifications():
    original = [["a"], ["b"], ["c"]]
    copy = duplicate_schedule(original)
    add_event(copy, 0, "x")
    add_event(copy, 2, "y")
    assert original == [["a"], ["b"], ["c"]]
    assert copy == [["a", "x"], ["b"], ["c", "y"]]
