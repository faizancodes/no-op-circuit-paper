from file_validator import validate_file_extension

def test_lowercase_extensions():
    assert validate_file_extension("document.pdf") == True
    assert validate_file_extension("image.jpg") == True
    assert validate_file_extension("notes.txt") == True

def test_uppercase_extensions():
    assert validate_file_extension("report.PDF") == True
    assert validate_file_extension("photo.PNG") == True

def test_mixed_case_extensions():
    assert validate_file_extension("file.PdF") == True
    assert validate_file_extension("picture.JpG") == True

def test_invalid_extensions():
    assert validate_file_extension("script.py") == False
    assert validate_file_extension("data.csv") == False

def test_no_extension():
    assert validate_file_extension("noextension") == False
    assert validate_file_extension("") == False
