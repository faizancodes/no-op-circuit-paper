from library import sort_books_by_year

def test_first_result_newest_book():
    books = [
        {'title': 'Old Book', 'year': 1990},
        {'title': 'New Book', 'year': 2020},
        {'title': 'Middle Book', 'year': 2005}
    ]
    result = sort_books_by_year(books)
    assert result[0]['year'] == 2020

def test_descending_order():
    books = [
        {'title': 'Book A', 'year': 2000},
        {'title': 'Book B', 'year': 2015},
        {'title': 'Book C', 'year': 1995}
    ]
    result = sort_books_by_year(books)
    assert [b['year'] for b in result] == [2015, 2000, 1995]

def test_single_book():
    books = [{'title': 'Solo', 'year': 2010}]
    result = sort_books_by_year(books)
    assert result == [{'title': 'Solo', 'year': 2010}]

def test_empty_list():
    assert sort_books_by_year([]) == []
