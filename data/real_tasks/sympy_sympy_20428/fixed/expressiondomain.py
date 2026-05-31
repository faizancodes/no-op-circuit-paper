            else:
                return NotImplemented

        def __rsub__(f, g):
            return f.simplify(f.__class__(g).ex - f.ex)

        def __mul__(f, g):
            g = f._to_ex(g)

            if g is not None:
                return f.simplify(f.ex*g.ex)
            else:
                return NotImplemented

        def __rmul__(f, g):
            return f.simplify(f.__class__(g).ex*f.ex)

        def __pow__(f, n):
            n = f._to_ex(n)

            if n is not None:
                return f.simplify(f.ex**n.ex)
            else:
                return NotImplemented

        def __truediv__(f, g):
            g = f._to_ex(g)

            if g is not None:
                return f.simplify(f.ex/g.ex)
            else:
                return NotImplemented

        def __rtruediv__(f, g):
            return f.simplify(f.__class__(g).ex/f.ex)

        def __eq__(f, g):
            return f.ex == f.__class__(g).ex

        def __ne__(f, g):
            return not f == g

        def __bool__(f):
            return not f.ex.is_zero

        def gcd(f, g):
            from sympy.polys import gcd
            return f.__class__(gcd(f.ex, f.__class__(g).ex))

        def lcm(f, g):
            from sympy.polys import lcm
            return f.__class__(lcm(f.ex, f.__class__(g).ex))

    dtype = Expression

    zero = Expression(0)
    one = Expression(1)

    rep = 'EX'

    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self):
        pass

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        return self.dtype(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ_gmpy(K1, a, K0):
