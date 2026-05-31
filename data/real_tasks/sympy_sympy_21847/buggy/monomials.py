    """
    n = len(variables)
    if is_sequence(max_degrees):
        if len(max_degrees) != n:
            raise ValueError('Argument sizes do not match')
        if min_degrees is None:
            min_degrees = [0]*n
        elif not is_sequence(min_degrees):
            raise ValueError('min_degrees is not a list')
        else:
            if len(min_degrees) != n:
                raise ValueError('Argument sizes do not match')
            if any(i < 0 for i in min_degrees):
                raise ValueError("min_degrees can't contain negative numbers")
        total_degree = False
    else:
        max_degree = max_degrees
        if max_degree < 0:
            raise ValueError("max_degrees can't be negative")
        if min_degrees is None:
            min_degree = 0
        else:
            if min_degrees < 0:
                raise ValueError("min_degrees can't be negative")
            min_degree = min_degrees
        total_degree = True
    if total_degree:
        if min_degree > max_degree:
            return
        if not variables or max_degree == 0:
            yield S.One
            return
        # Force to list in case of passed tuple or other incompatible collection
        variables = list(variables) + [S.One]
        if all(variable.is_commutative for variable in variables):
            monomials_list_comm = []
            for item in combinations_with_replacement(variables, max_degree):
                powers = dict()
                for variable in variables:
                    powers[variable] = 0
                for variable in item:
                    if variable != 1:
                        powers[variable] += 1
                if max(powers.values()) >= min_degree:
                    monomials_list_comm.append(Mul(*item))
            yield from set(monomials_list_comm)
        else:
            monomials_list_non_comm = []
            for item in product(variables, repeat=max_degree):
                powers = dict()
                for variable in variables:
                    powers[variable] = 0
                for variable in item:
                    if variable != 1:
                        powers[variable] += 1
                if max(powers.values()) >= min_degree:
                    monomials_list_non_comm.append(Mul(*item))
            yield from set(monomials_list_non_comm)
    else:
        if any(min_degrees[i] > max_degrees[i] for i in range(n)):
            raise ValueError('min_degrees[i] must be <= max_degrees[i] for all i')
        power_lists = []
        for var, min_d, max_d in zip(variables, min_degrees, max_degrees):
            power_lists.append([var**i for i in range(min_d, max_d + 1)])
        for powers in product(*power_lists):
            yield Mul(*powers)

def monomial_count(V, N):
    r"""
    Computes the number of monomials.

    The number of monomials is given by the following formula:

    .. math::

        \frac{(\#V + N)!}{\#V! N!}

    where `N` is a total degree and `V` is a set of variables.

    Examples
    ========

    >>> from sympy.polys.monomials import itermonomials, monomial_count
    >>> from sympy.polys.orderings import monomial_key
    >>> from sympy.abc import x, y

    >>> monomial_count(2, 2)
