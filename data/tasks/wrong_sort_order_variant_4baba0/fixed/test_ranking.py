from ranking import rank_students

def test_highest_score_first():
    students = [
        {'name': 'Alice', 'score': 85},
        {'name': 'Bob', 'score': 92},
        {'name': 'Charlie', 'score': 78}
    ]
    result = rank_students(students)
    assert result[0]['name'] == 'Bob'
    assert result[0]['score'] == 92

def test_two_students():
    students = [
        {'name': 'Dave', 'score': 60},
        {'name': 'Eve', 'score': 95}
    ]
    result = rank_students(students)
    assert result[0]['score'] == 95
    assert result[1]['score'] == 60

def test_single_student():
    students = [{'name': 'Frank', 'score': 88}]
    result = rank_students(students)
    assert len(result) == 1
    assert result[0]['name'] == 'Frank'

def test_descending_order_maintained():
    students = [
        {'name': 'Grace', 'score': 100},
        {'name': 'Hank', 'score': 90},
        {'name': 'Ivy', 'score': 80},
        {'name': 'Jack', 'score': 70}
    ]
    result = rank_students(students)
    scores = [s['score'] for s in result]
    assert scores == [100, 90, 80, 70]
