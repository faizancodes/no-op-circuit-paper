    1 0 0 zoo
    sqrt(3)/2 1/2 sqrt(3)/3 sqrt(3)
    sqrt(2)/2 sqrt(2)/2 1 1
    1/2 sqrt(3)/2 sqrt(3) sqrt(3)/3
    0 1 zoo 0
    """
    # special values at 0, pi/6, pi/4, pi/3, pi/2 already handled
    return rv


def _TR56(rv, f, g, h, max, pow):
    """Helper for TR5 and TR6 to replace f**2 with h(g**2)

    Options
    =======

    max :   controls size of exponent that can appear on f
            e.g. if max=4 then f**4 will be changed to h(g**2)**2.
    pow :   controls whether the exponent must be a perfect power of 2
            e.g. if pow=True (and max >= 6) then f**6 will not be changed
            but f**8 will be changed to h(g**2)**4

    >>> from sympy.simplify.fu import _TR56 as T
    >>> from sympy.abc import x
    >>> from sympy import sin, cos
    >>> h = lambda x: 1 - x
    >>> T(sin(x)**3, sin, cos, h, 4, False)
    sin(x)**3
    >>> T(sin(x)**6, sin, cos, h, 6, False)
    (1 - cos(x)**2)**3
    >>> T(sin(x)**6, sin, cos, h, 6, True)
    sin(x)**6
    >>> T(sin(x)**8, sin, cos, h, 10, True)
    (1 - cos(x)**2)**4
    """

    def _f(rv):
        # I'm not sure if this transformation should target all even powers
        # or only those expressible as powers of 2. Also, should it only
        # make the changes in powers that appear in sums -- making an isolated
        # change is not going to allow a simplification as far as I can tell.
        if not (rv.is_Pow and rv.base.func == f):
            return rv

        if (rv.exp < 0) == True:
            return rv
        if (rv.exp > max) == True:
            return rv
        if rv.exp == 2:
            return h(g(rv.base.args[0])**2)
        else:
            if rv.exp == 4:
                e = 2
            elif not pow:
                if rv.exp % 2:
                    return rv
                e = rv.exp//2
            else:
                p = perfect_power(rv.exp)
                if not p:
                    return rv
                e = rv.exp//2
            return h(g(rv.base.args[0])**2)**e

    return bottom_up(rv, _f)


def TR5(rv, max=4, pow=False):
    """Replacement of sin**2 with 1 - cos(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR5
    >>> from sympy.abc import x
    >>> from sympy import sin
    >>> TR5(sin(x)**2)
    1 - cos(x)**2
    >>> TR5(sin(x)**-2)  # unchanged
    sin(x)**(-2)
    >>> TR5(sin(x)**4)
    (1 - cos(x)**2)**2
    """
    return _TR56(rv, sin, cos, lambda x: 1 - x, max=max, pow=pow)
