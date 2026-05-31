from auth import check_user_role

def test_exact_match():
    assert check_user_role("admin", "admin") == True

def test_different_roles():
    assert check_user_role("user", "admin") == False

def test_capitalized_user_role():
    assert check_user_role("Admin", "admin") == True

def test_uppercase_user_role():
    assert check_user_role("MODERATOR", "moderator") == True

def test_mixed_case_both():
    assert check_user_role("EdItOr", "EDITOR") == True
