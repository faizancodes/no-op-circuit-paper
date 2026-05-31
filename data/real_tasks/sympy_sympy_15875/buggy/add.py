                if a.is_zero:
                    pass
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im_I.append(a*S.ImaginaryUnit)
            elif (S.ImaginaryUnit*a).is_real:
                im_I.append(a*S.ImaginaryUnit)
            else:
                return
        b = self.func(*nz)
        if b.is_zero:
            return fuzzy_not(self.func(*im_I).is_zero)
        elif b.is_zero is False:
            return False

    def _eval_is_zero(self):
        if self.is_commutative is False:
            # issue 10528: there is no way to know if a nc symbol
            # is zero or not
            return
        nz = []
        z = 0
        im_or_z = False
        im = False
        for a in self.args:
            if a.is_real:
                if a.is_zero:
                    z += 1
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im = True
            elif (S.ImaginaryUnit*a).is_real:
                im_or_z = True
            else:
                return
        if z == len(self.args):
            return True
        if len(nz) == len(self.args):
            return None
        b = self.func(*nz)
        if b.is_zero:
            if not im_or_z and not im:
                return True
            if im and not im_or_z:
                return False
        if b.is_zero is False:
            return False

    def _eval_is_odd(self):
        l = [f for f in self.args if not (f.is_even is True)]
        if not l:
            return False
        if l[0].is_odd:
            return self._new_rawargs(*l[1:]).is_even

    def _eval_is_irrational(self):
        for t in self.args:
            a = t.is_irrational
            if a:
                others = list(self.args)
                others.remove(t)
                if all(x.is_rational is True for x in others):
                    return True
                return None
            if a is None:
                return
        return False

    def _eval_is_positive(self):
        from sympy.core.exprtools import _monotonic_sign
        if self.is_number:
            return super(Add, self)._eval_is_positive()
        c, a = self.as_coeff_Add()
        if not c.is_zero:
            v = _monotonic_sign(a)
            if v is not None:
                s = v + c
                if s != self and s.is_positive and a.is_nonnegative:
                    return True
                if len(self.free_symbols) == 1:
                    v = _monotonic_sign(self)
