from titlecase import convert_to_title_case

def test_empty_string():
    assert convert_to_title_case("") == ""

def test_single_word_lowercase():
    assert convert_to_title_case("hello") == "Hello"

def test_single_word_uppercase():
    assert convert_to_title_case("WORLD") == "World"

def test_multiple_words_mixed_case():
    assert convert_to_title_case("hello WORLD from PYTHON") == "Hello World From Python"

def test_already_title_case():
    assert convert_to_title_case("Title Case Text") == "Title Case Text"
