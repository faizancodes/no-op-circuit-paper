from config_merger import merge_config

def test_simple_override():
    base = {"timeout": 30, "retries": 3}
    overrides = {"timeout": 60}
    result = merge_config(base, overrides)
    assert result == {"timeout": 60, "retries": 3}
    assert base == {"timeout": 30, "retries": 3}

def test_nested_merge():
    base = {"database": {"host": "localhost", "port": 5432}}
    overrides = {"database": {"port": 3306}}
    result = merge_config(base, overrides)
    assert result == {"database": {"host": "localhost", "port": 3306}}

def test_result_independent_from_base():
    base = {"server": {"host": "localhost", "port": 8000}}
    overrides = {"server": {"port": 9000}}
    result = merge_config(base, overrides)
    result["server"]["host"] = "example.com"
    assert base["server"]["host"] == "localhost"

def test_add_new_key():
    base = {"timeout": 30}
    overrides = {"retries": 5}
    result = merge_config(base, overrides)
    assert result == {"timeout": 30, "retries": 5}
