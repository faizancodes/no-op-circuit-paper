from inventory import create_inventory_copy, add_item_to_category

def test_copy_preserves_structure():
    original = {"weapons": ["sword", "bow"], "potions": ["health"]}
    copied = create_inventory_copy(original)
    assert copied == original
    assert copied is not original

def test_modifying_copy_does_not_affect_original():
    original = {"weapons": ["sword", "bow"], "potions": ["health"]}
    copied = create_inventory_copy(original)
    add_item_to_category(copied, "weapons", "axe")
    assert "axe" in copied["weapons"]
    assert "axe" not in original["weapons"]
    assert len(original["weapons"]) == 2

def test_adding_new_category_to_copy():
    original = {"weapons": ["sword"]}
    copied = create_inventory_copy(original)
    add_item_to_category(copied, "armor", "shield")
    assert "armor" in copied
    assert "armor" not in original

def test_empty_inventory_copy():
    original = {}
    copied = create_inventory_copy(original)
    assert copied == {}
    add_item_to_category(copied, "items", "key")
    assert "items" not in original
