            'log': 11,
            'sin': 20,
            'cos': 21,
            'tan': 22,
            'cot': 23,
            'sinh': 30,
            'cosh': 31,
            'tanh': 32,
            'coth': 33,
            'conjugate': 40,
            're': 41,
            'im': 42,
            'arg': 43,
        }
        name = cls.__name__

        try:
            i = funcs[name]
        except KeyError:
            i = 0 if isinstance(cls.nargs, Naturals0) else 10000

        return 4, i, name

    @property
    def is_commutative(self):
        """
        Returns whether the functon is commutative.
        """
        if all(getattr(t, 'is_commutative') for t in self.args):
            return True
        else:
            return False

    def _eval_evalf(self, prec):
        # Lookup mpmath function based on name
        fname = self.func.__name__
        try:
            if not hasattr(mpmath, fname):
                from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
                fname = MPMATH_TRANSLATIONS[fname]
            func = getattr(mpmath, fname)
        except (AttributeError, KeyError):
            try:
                return Float(self._imp_(*self.args), prec)
            except (AttributeError, TypeError, ValueError):
                return

        # Convert all args to mpf or mpc
        # Convert the arguments to *higher* precision than requested for the
        # final result.
        # XXX + 5 is a guess, it is similar to what is used in evalf.py. Should
        #     we be more intelligent about it?
        try:
            args = [arg._to_mpmath(prec + 5) for arg in self.args]
            def bad(m):
                from mpmath import mpf, mpc
                # the precision of an mpf value is the last element
                # if that is 1 (and m[1] is not 1 which would indicate a
                # power of 2), then the eval failed; so check that none of
                # the arguments failed to compute to a finite precision.
                # Note: An mpc value has two parts, the re and imag tuple;
                # check each of those parts, too. Anything else is allowed to
                # pass
                if isinstance(m, mpf):
                    m = m._mpf_
                    return m[1] !=1 and m[-1] == 1
                elif isinstance(m, mpc):
                    m, n = m._mpc_
                    return m[1] !=1 and m[-1] == 1 and \
                        n[1] !=1 and n[-1] == 1
                else:
                    return False
            if any(bad(a) for a in args):
                raise ValueError  # one or more args failed to compute with significance
        except ValueError:
            return

        with mpmath.workprec(prec):
            v = func(*args)

        return Expr._from_mpmath(v, prec)

    def _eval_derivative(self, s):
        # f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
        i = 0
        l = []
        for a in self.args:
