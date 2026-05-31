from sanitizer import sanitize_filename

def test_valid_filename_unchanged():
    assert sanitize_filename('report.txt') == 'report.txt'

def test_removes_invalid_chars():
    assert sanitize_filename('file:name?.txt') == 'filename.txt'

def test_multiple_invalid_chars():
    assert sanitize_filename('bad<file>name|here*.doc') == 'badfilenamehere.doc'

def test_empty_string():
    assert sanitize_filename('') == ''

def test_only_valid_chars():
    assert sanitize_filename('my_document-2024.pdf') == 'my_document-2024.pdf'
