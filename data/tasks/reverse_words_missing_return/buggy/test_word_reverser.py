from word_reverser import reverse_words

def test_empty_string():
    assert reverse_words("") == ""

def test_single_word():
    assert reverse_words("hello") == "olleh"

def test_multiple_words():
    assert reverse_words("hello world") == "olleh dlrow"

def test_mixed_case():
    assert reverse_words("Python Code") == "nohtyP edoC"

def test_multiple_spaces():
    assert reverse_words("a  b") == "a  b"
