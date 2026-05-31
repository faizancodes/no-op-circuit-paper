from color_validator import validate_color_name

def test_lowercase_colors_valid():
    assert validate_color_name('red') == True
    assert validate_color_name('blue') == True
    assert validate_color_name('green') == True

def test_uppercase_colors_valid():
    assert validate_color_name('RED') == True
    assert validate_color_name('BLUE') == True

def test_mixed_case_colors_valid():
    assert validate_color_name('Red') == True
    assert validate_color_name('GrEeN') == True

def test_invalid_colors():
    assert validate_color_name('notacolor') == False
    assert validate_color_name('') == False
