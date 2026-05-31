from action_log import register_action

def test_single_action_default_log():
    result = register_action('login')
    assert result == ['login']

def test_multiple_independent_calls():
    first = register_action('start')
    second = register_action('stop')
    assert first == ['start']
    assert second == ['stop']

def test_custom_log_provided():
    my_log = ['init']
    result = register_action('process', my_log)
    assert result == ['init', 'process']
    assert my_log == ['init', 'process']

def test_multiple_actions_same_custom_log():
    shared_log = []
    register_action('open', shared_log)
    register_action('read', shared_log)
    register_action('close', shared_log)
    assert shared_log == ['open', 'read', 'close']
