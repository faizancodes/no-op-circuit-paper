            else:
                numer.append(term)
        elif term.is_Rational:
            n, d = term.as_numer_denom()
            numer.append(n)
            denom.append(d)
        else:
            numer.append(term)
    if exact:
        return Mul(*numer, evaluate=False), Mul(*denom, evaluate=False)
    else:
        return Mul(*numer), Mul(*denom)


def numer(expr):
    return fraction(expr)[0]


def denom(expr):
    return fraction(expr)[1]


def fraction_expand(expr, **hints):
    return expr.expand(frac=True, **hints)


def numer_expand(expr, **hints):
    a, b = fraction(expr)
    return a.expand(numer=True, **hints) / b


def denom_expand(expr, **hints):
    a, b = fraction(expr)
    return a / b.expand(denom=True, **hints)


expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand


def split_surds(expr):
    """
    split an expression with terms whose squares are rationals
    into a sum of terms whose surds squared have gcd equal to g
    and a sum of terms with surds squared prime with g

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import split_surds
    >>> split_surds(3*sqrt(3) + sqrt(5)/7 + sqrt(6) + sqrt(10) + sqrt(15))
    (3, sqrt(2) + sqrt(5) + 3, sqrt(5)/7 + sqrt(10))
    """
    args = sorted(expr.args, key=default_sort_key)
    coeff_muls = [x.as_coeff_Mul() for x in args]
    surds = [x[1]**2 for x in coeff_muls if x[1].is_Pow]
    surds.sort(key=default_sort_key)
    g, b1, b2 = _split_gcd(*surds)
    g2 = g
    if not b2 and len(b1) >= 2:
        b1n = [x/g for x in b1]
        b1n = [x for x in b1n if x != 1]
        # only a common factor has been factored; split again
        g1, b1n, b2 = _split_gcd(*b1n)
        g2 = g*g1
    a1v, a2v = [], []
    for c, s in coeff_muls:
        if s.is_Pow and s.exp == S.Half:
            s1 = s.base
            if s1 in b1:
                a1v.append(c*sqrt(s1/g2))
            else:
                a2v.append(c*s)
        else:
            a2v.append(c*s)
    a = Add(*a1v)
    b = Add(*a2v)
    return g2, a, b


def _split_gcd(*a):
    """
    split the list of integers ``a`` into a list of integers, ``a1`` having
    ``g = gcd(a1)``, and a list ``a2`` whose elements are not divisible by
    ``g``.  Returns ``g, a1, a2``

    Examples
