def dup_ext_factor(f, K):
    """Factor univariate polynomials over algebraic number fields. """
    n, lc = dup_degree(f), dup_LC(f, K)

    f = dup_monic(f, K)

    if n <= 0:
        return lc, []
    if n == 1:
        return lc, [(f, 1)]

    f, F = dup_sqf_part(f, K), f
    s, g, r = dup_sqf_norm(f, K)

    factors = dup_factor_list_include(r, K.dom)

    if len(factors) == 1:
        return lc, [(f, n//dup_degree(f))]

    H = s*K.unit

    for i, (factor, _) in enumerate(factors):
        h = dup_convert(factor, K.dom, K)
        h, _, g = dup_inner_gcd(h, g, K)
        h = dup_shift(h, H, K)
        factors[i] = h

    factors = dup_trial_division(F, factors, K)
    return lc, factors


def dmp_ext_factor(f, u, K):
    """Factor multivariate polynomials over algebraic number fields. """
    if not u:
        return dup_ext_factor(f, K)

    lc = dmp_ground_LC(f, u, K)
    f = dmp_ground_monic(f, u, K)

    if all(d <= 0 for d in dmp_degree_list(f, u)):
        return lc, []

    f, F = dmp_sqf_part(f, u, K), f
    s, g, r = dmp_sqf_norm(f, u, K)

    factors = dmp_factor_list_include(r, u, K.dom)

    if len(factors) == 1:
        factors = [f]
    else:
        H = dmp_raise([K.one, s*K.unit], u, 0, K)

        for i, (factor, _) in enumerate(factors):
            h = dmp_convert(factor, u, K.dom, K)
            h, _, g = dmp_inner_gcd(h, g, u, K)
            h = dmp_compose(h, H, u, K)
            factors[i] = h

    return lc, dmp_trial_division(F, factors, u, K)


def dup_gf_factor(f, K):
    """Factor univariate polynomials over finite fields. """
    f = dup_convert(f, K, K.dom)

    coeff, factors = gf_factor(f, K.mod, K.dom)

    for i, (f, k) in enumerate(factors):
        factors[i] = (dup_convert(f, K.dom, K), k)

    return K.convert(coeff, K.dom), factors


def dmp_gf_factor(f, u, K):
    """Factor multivariate polynomials over finite fields. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_factor_list(f, K0):
    """Factor univariate polynomials into irreducibles in `K[x]`. """
    j, f = dup_terms_gcd(f, K0)
    cont, f = dup_primitive(f, K0)

    if K0.is_FiniteField:
        coeff, factors = dup_gf_factor(f, K0)
    elif K0.is_Algebraic:
        coeff, factors = dup_ext_factor(f, K0)
