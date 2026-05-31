from text_extractor import extract_middle

def test_single_delimiter():
    assert extract_middle('|') == '|'

def test_two_delimiters_with_content():
    assert extract_middle('|hello|') == '|hello|'

def test_multiple_delimiters():
    assert extract_middle('start |a| middle |b| end') == '|a| middle |b|'

def test_no_delimiter():
    assert extract_middle('no delimiter here') == ''

def test_delimiter_at_boundaries():
    assert extract_middle('|x|', '|') == '|x|'
