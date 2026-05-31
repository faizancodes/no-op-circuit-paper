from tracker import track_event, get_session_events

def test_single_event():
    result = track_event('login')
    assert result == ['login']

def test_multiple_events_with_explicit_history():
    session = []
    result1 = track_event('login', session)
    result2 = track_event('view_page', session)
    assert result2 == ['login', 'view_page']

def test_independent_calls_should_not_share_state():
    first = track_event('signup')
    second = track_event('logout')
    assert first == ['signup']
    assert second == ['logout']

def test_get_session_events_returns_copy():
    events = ['click', 'scroll']
    copy = get_session_events(events)
    copy.append('new_event')
    assert events == ['click', 'scroll']
    assert copy == ['click', 'scroll', 'new_event']
