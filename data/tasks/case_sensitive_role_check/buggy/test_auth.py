from auth import check_permission

def test_admin_has_admin_permission():
    assert check_permission("admin", "admin") == True

def test_admin_has_moderator_permission():
    assert check_permission("admin", "moderator") == True

def test_user_lacks_admin_permission():
    assert check_permission("user", "admin") == False

def test_case_insensitive_user_role():
    assert check_permission("Admin", "moderator") == True

def test_case_insensitive_both_roles():
    assert check_permission("MODERATOR", "User") == True
