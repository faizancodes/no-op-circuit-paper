from products import sort_products_by_price

def test_most_expensive_first():
    products = [
        {'name': 'Laptop', 'price': 999},
        {'name': 'Mouse', 'price': 25},
        {'name': 'Keyboard', 'price': 75}
    ]
    result = sort_products_by_price(products)
    assert result[0]['name'] == 'Laptop'
    assert result[0]['price'] == 999

def test_descending_order():
    products = [
        {'name': 'A', 'price': 10},
        {'name': 'B', 'price': 50},
        {'name': 'C', 'price': 30}
    ]
    result = sort_products_by_price(products)
    assert [p['price'] for p in result] == [50, 30, 10]

def test_single_product():
    products = [{'name': 'Item', 'price': 100}]
    result = sort_products_by_price(products)
    assert result == products

def test_equal_prices():
    products = [
        {'name': 'X', 'price': 20},
        {'name': 'Y', 'price': 20}
    ]
    result = sort_products_by_price(products)
    assert len(result) == 2
    assert all(p['price'] == 20 for p in result)
