
        if k not in term.free_symbols:
            if (term - 1).is_zero:
                return S.One
            return term**(n - a + 1)

        if a == n:
            return term.subs(k, a)

        if term.has(KroneckerDelta) and _has_simple_delta(term, limits[0]):
            return deltaproduct(term, limits)

        dif = n - a
        if dif.is_Integer:
            return Mul(*[term.subs(k, a + i) for i in range(dif + 1)])

        elif term.is_polynomial(k):
            poly = term.as_poly(k)

            A = B = Q = S.One

            all_roots = roots(poly)

            M = 0
            for r, m in all_roots.items():
                M += m
                A *= RisingFactorial(a - r, n - a + 1)**m
                Q *= (n - r)**m

            if M < poly.degree():
                arg = quo(poly, Q.as_poly(k))
                B = self.func(arg, (k, a, n)).doit()

            return poly.LC()**(n - a + 1) * A * B

        elif term.is_Add:
            p, q = term.as_numer_denom()
            q = self._eval_product(q, (k, a, n))
            if q.is_Number:

                # There is expression, which couldn't change by
                # as_numer_denom(). E.g. n**(2/3) + 1 --> (n**(2/3) + 1, 1).
                # We have to catch this case.

                p = sum([self._eval_product(i, (k, a, n)) for i in p.as_coeff_Add()])
            else:
                p = self._eval_product(p, (k, a, n))
            return p / q

        elif term.is_Mul:
            exclude, include = [], []

            for t in term.args:
                p = self._eval_product(t, (k, a, n))

                if p is not None:
                    exclude.append(p)
                else:
                    include.append(t)

            if not exclude:
                return None
            else:
                arg = term._new_rawargs(*include)
                A = Mul(*exclude)
                B = self.func(arg, (k, a, n)).doit()
                return A * B

        elif term.is_Pow:
            if not term.base.has(k):
                s = summation(term.exp, (k, a, n))

                return term.base**s
            elif not term.exp.has(k):
                p = self._eval_product(term.base, (k, a, n))

                if p is not None:
                    return p**term.exp

        elif isinstance(term, Product):
            evaluated = term.doit()
            f = self._eval_product(evaluated, limits)
            if f is None:
                return self.func(evaluated, limits)
            else:
                return f

    def _eval_simplify(self, ratio, measure):
