from ranking import rank_competitors

def test_highest_scorer_first():
    competitors = [
        {'name': 'Alice', 'score': 850},
        {'name': 'Bob', 'score': 950},
        {'name': 'Charlie', 'score': 720}
    ]
    result = rank_competitors(competitors)
    assert result[0]['name'] == 'Bob'
    assert result[0]['score'] == 950

def test_lowest_scorer_last():
    competitors = [
        {'name': 'Alice', 'score': 850},
        {'name': 'Bob', 'score': 950},
        {'name': 'Charlie', 'score': 720}
    ]
    result = rank_competitors(competitors)
    assert result[-1]['name'] == 'Charlie'
    assert result[-1]['score'] == 720

def test_single_competitor():
    competitors = [{'name': 'Solo', 'score': 500}]
    result = rank_competitors(competitors)
    assert len(result) == 1
    assert result[0]['name'] == 'Solo'

def test_empty_list():
    assert rank_competitors([]) == []
