from config_manager import apply_overrides

def test_simple_override():
    base = {"port": 8080, "host": "localhost"}
    overrides = {"port": 9000}
    result = apply_overrides(base, overrides)
    assert result["port"] == 9000
    assert result["host"] == "localhost"
    assert base["port"] == 8080

def test_nested_list_independence():
    base = {"server": {"allowed_ips": ["127.0.0.1", "192.168.1.1"]}}
    overrides = {"timeout": 30}
    result = apply_overrides(base, overrides)
    result["server"]["allowed_ips"].append("10.0.0.1")
    assert len(base["server"]["allowed_ips"]) == 2
    assert len(result["server"]["allowed_ips"]) == 3

def test_nested_dict_independence():
    base = {"logging": {"handlers": {"file": "/var/log/app.log"}}}
    overrides = {"debug": True}
    result = apply_overrides(base, overrides)
    result["logging"]["handlers"]["console"] = "stdout"
    assert "console" not in base["logging"]["handlers"]
    assert "console" in result["logging"]["handlers"]

def test_override_replaces_value():
    base = {"cache": {"size": 100}, "workers": 4}
    overrides = {"cache": {"size": 200, "ttl": 3600}}
    result = apply_overrides(base, overrides)
    assert result["cache"]["size"] == 200
    assert result["cache"]["ttl"] == 3600
    assert base["cache"] == {"size": 100}
