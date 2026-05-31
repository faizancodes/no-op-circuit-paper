    The probability distribution function depends on a single parameter
    `n` which is an integer.

    The density of the Irwin-Hall distribution is given by

    .. math ::
        f(x) := \frac{1}{(n-1)!}\sum_{k=0}^{\lfloor x\rfloor}(-1)^k
                \binom{n}{k}(x-k)^{n-1}

    Parameters
    ==========

    n : A positive Integer, `n > 0`

    Returns
    =======

    A RandomSymbol.

    Examples
    ========

    >>> from sympy.stats import UniformSum, density
    >>> from sympy import Symbol, pprint

    >>> n = Symbol("n", integer=True)
    >>> z = Symbol("z")

    >>> X = UniformSum("x", n)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    floor(z)
      ___
      \  `
       \         k         n - 1 /n\
        )    (-1) *(-k + z)     *| |
       /                         \k/
      /__,
     k = 0
    --------------------------------
                (n - 1)!

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Uniform_sum_distribution
    .. [2] http://mathworld.wolfram.com/UniformSumDistribution.html
    """

    return rv(name, UniformSumDistribution, (n, ))

#-------------------------------------------------------------------------------
# VonMises distribution --------------------------------------------------------


class VonMisesDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'k')

    set = Interval(0, 2*pi)

    @staticmethod
    def check(mu, k):
        _value_check(k > 0, "k must be positive")

    def pdf(self, x):
        mu, k = self.mu, self.k
        return exp(k*cos(x-mu)) / (2*pi*besseli(0, k))


def VonMises(name, mu, k):
    r"""
    Create a Continuous Random Variable with a von Mises distribution.

    The density of the von Mises distribution is given by

    .. math::
        f(x) := \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    with :math:`x \in [0,2\pi]`.

    Parameters
    ==========

    mu : Real number, measure of location
    k : Real number, measure of concentration
