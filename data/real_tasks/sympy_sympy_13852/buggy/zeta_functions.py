
    For :math:`z \in \{0, 1, -1\}`, the polylogarithm is automatically expressed
    using other functions:

    >>> from sympy import polylog
    >>> from sympy.abc import s
    >>> polylog(s, 0)
    0
    >>> polylog(s, 1)
    zeta(s)
    >>> polylog(s, -1)
    -dirichlet_eta(s)

    If :math:`s` is a negative integer, :math:`0` or :math:`1`, the
    polylogarithm can be expressed using elementary functions. This can be
    done using expand_func():

    >>> from sympy import expand_func
    >>> from sympy.abc import z
    >>> expand_func(polylog(1, z))
    -log(z*exp_polar(-I*pi) + 1)
    >>> expand_func(polylog(0, z))
    z/(-z + 1)

    The derivative with respect to :math:`z` can be computed in closed form:

    >>> polylog(s, z).diff(z)
    polylog(s - 1, z)/z

    The polylogarithm can be expressed in terms of the lerch transcendent:

    >>> from sympy import lerchphi
    >>> polylog(s, z).rewrite(lerchphi)
    z*lerchphi(z, s, 1)
    """

    @classmethod
    def eval(cls, s, z):
        if z == 1:
            return zeta(s)
        elif z == -1:
            return -dirichlet_eta(s)
        elif z == 0:
            return 0

    def fdiff(self, argindex=1):
        s, z = self.args
        if argindex == 2:
            return polylog(s - 1, z)/z
        raise ArgumentIndexError

    def _eval_rewrite_as_lerchphi(self, s, z):
        return z*lerchphi(z, s, 1)

    def _eval_expand_func(self, **hints):
        from sympy import log, expand_mul, Dummy, exp_polar, I
        s, z = self.args
        if s == 1:
            return -log(1 + exp_polar(-I*pi)*z)
        if s.is_Integer and s <= 0:
            u = Dummy('u')
            start = u/(1 - u)
            for _ in range(-s):
                start = u*start.diff(u)
            return expand_mul(start).subs(u, z)
        return polylog(s, z)

###############################################################################
###################### HURWITZ GENERALIZED ZETA FUNCTION ######################
###############################################################################


class zeta(Function):
    r"""
    Hurwitz zeta function (or Riemann zeta function).

    For `\operatorname{Re}(a) > 0` and `\operatorname{Re}(s) > 1`, this function is defined as

    .. math:: \zeta(s, a) = \sum_{n=0}^\infty \frac{1}{(n + a)^s},

    where the standard choice of argument for :math:`n + a` is used. For fixed
    :math:`a` with `\operatorname{Re}(a) > 0` the Hurwitz zeta function admits a
    meromorphic continuation to all of :math:`\mathbb{C}`, it is an unbranched
    function with a simple pole at :math:`s = 1`.

    Analytic continuation to other :math:`a` is possible under some circumstances,
    but this is not typically done.
