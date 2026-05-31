from loan import calculate_monthly_payment
import math

def test_zero_interest_rate():
    payment = calculate_monthly_payment(12000, 0, 12)
    assert payment == 1000.0

def test_standard_loan():
    payment = calculate_monthly_payment(10000, 0.06, 12)
    assert math.isclose(payment, 860.66, rel_tol=0.01)

def test_longer_term_loan():
    payment = calculate_monthly_payment(200000, 0.045, 360)
    assert math.isclose(payment, 1013.37, rel_tol=0.01)

def test_short_term_high_rate():
    payment = calculate_monthly_payment(5000, 0.12, 6)
    assert math.isclose(payment, 861.45, rel_tol=0.01)
