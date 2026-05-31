
    def __mod__(f, g):
        return f.rem(g)

    def __floordiv__(f, g):
        if isinstance(g, DMP):
            return f.quo(g)
        else:
            try:
                return f.quo_ground(g)
            except TypeError:
                return NotImplemented

    def __eq__(f, g):
        try:
            _, _, _, F, G = f.unify(g)

            if f.lev == g.lev:
                return F == G
        except UnificationFailed:
            pass

        return False

    def __ne__(f, g):
        return not f.__eq__(g)

    def eq(f, g, strict=False):
        if not strict:
            return f.__eq__(g)
        else:
            return f._strict_eq(g)

    def ne(f, g, strict=False):
        return not f.eq(g, strict=strict)

    def _strict_eq(f, g):
        return isinstance(g, f.__class__) and f.lev == g.lev \
            and f.dom == g.dom \
            and f.rep == g.rep

    def __lt__(f, g):
        _, _, _, F, G = f.unify(g)
        return F < G

    def __le__(f, g):
        _, _, _, F, G = f.unify(g)
        return F <= G

    def __gt__(f, g):
        _, _, _, F, G = f.unify(g)
        return F > G

    def __ge__(f, g):
        _, _, _, F, G = f.unify(g)
        return F >= G

    def __nonzero__(f):
        return not dmp_zero_p(f.rep, f.lev)

    __bool__ = __nonzero__


def init_normal_DMF(num, den, lev, dom):
    return DMF(dmp_normal(num, lev, dom),
               dmp_normal(den, lev, dom), dom, lev)


class DMF(PicklableWithSlots, CantSympify):
    """Dense Multivariate Fractions over `K`. """

    __slots__ = ['num', 'den', 'lev', 'dom', 'ring']

    def __init__(self, rep, dom, lev=None, ring=None):
        num, den, lev = self._parse(rep, dom, lev)
        num, den = dmp_cancel(num, den, lev, dom)

        self.num = num
        self.den = den
        self.lev = lev
        self.dom = dom
        self.ring = ring

    @classmethod
    def new(cls, rep, dom, lev=None, ring=None):
        num, den, lev = cls._parse(rep, dom, lev)

        obj = object.__new__(cls)

        obj.num = num
        obj.den = den
        obj.lev = lev
        obj.dom = dom
        obj.ring = ring

        return obj

    @classmethod
    def _parse(cls, rep, dom, lev=None):
