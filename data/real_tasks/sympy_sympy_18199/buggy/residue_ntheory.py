    Root of ``x**q = s mod p``, ``p`` prime and ``q`` divides ``p - 1``

    References
    ==========

    .. [1] A. M. Johnston "A Generalized qth Root Algorithm"

    """
    g = primitive_root(p)
    if not isprime(q):
        r = _nthroot_mod2(s, q, p)
    else:
        f = p - 1
        assert (p - 1) % q == 0
        # determine k
        k = 0
        while f % q == 0:
            k += 1
            f = f // q
        # find z, x, r1
        f1 = igcdex(-f, q)[0] % q
        z = f*f1
        x = (1 + z) // q
        r1 = pow(s, x, p)
        s1 = pow(s, f, p)
        h = pow(g, f*q, p)
        t = discrete_log(p, s1, h)
        g2 = pow(g, z*t, p)
        g3 = igcdex(g2, p)[0]
        r = r1*g3 % p
        #assert pow(r, q, p) == s
    res = [r]
    h = pow(g, (p - 1) // q, p)
    #assert pow(h, q, p) == 1
    hx = r
    for i in range(q - 1):
        hx = (hx*h) % p
        res.append(hx)
    if all_roots:
        res.sort()
        return res
    return min(res)


def nthroot_mod(a, n, p, all_roots=False):
    """
    Find the solutions to ``x**n = a mod p``

    Parameters
    ==========

    a : integer
    n : positive integer
    p : positive integer
    all_roots : if False returns the smallest root, else the list of roots

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import nthroot_mod
    >>> nthroot_mod(11, 4, 19)
    8
    >>> nthroot_mod(11, 4, 19, True)
    [8, 11]
    >>> nthroot_mod(68, 3, 109)
    23
    """
    from sympy.core.numbers import igcdex
    a, n, p = as_int(a), as_int(n), as_int(p)
    if n == 2:
        return sqrt_mod(a, p, all_roots)
    # see Hackman "Elementary Number Theory" (2009), page 76
    if not is_nthpow_residue(a, n, p):
        return None
    if not isprime(p):
        raise NotImplementedError("Not implemented for composite p")

    if (p - 1) % n == 0:
        return _nthroot_mod1(a, n, p, all_roots)
    # The roots of ``x**n - a = 0 (mod p)`` are roots of
    # ``gcd(x**n - a, x**(p - 1) - 1) = 0 (mod p)``
    pa = n
    pb = p - 1
    b = 1
    if pa < pb:
        a, pa, b, pb = b, pb, a, pa
