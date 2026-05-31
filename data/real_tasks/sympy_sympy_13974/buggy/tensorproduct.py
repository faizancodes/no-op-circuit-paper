    This is an example of the type of simplification that this function
    performs::

        >>> from sympy.physics.quantum.tensorproduct import \
                    tensor_product_simp_Mul, TensorProduct
        >>> from sympy import Symbol
        >>> A = Symbol('A',commutative=False)
        >>> B = Symbol('B',commutative=False)
        >>> C = Symbol('C',commutative=False)
        >>> D = Symbol('D',commutative=False)
        >>> e = TensorProduct(A,B)*TensorProduct(C,D)
        >>> e
        AxB*CxD
        >>> tensor_product_simp_Mul(e)
        (A*C)x(B*D)

    """
    # TODO: This won't work with Muls that have other composites of
    # TensorProducts, like an Add, Pow, Commutator, etc.
    # TODO: This only works for the equivalent of single Qbit gates.
    if not isinstance(e, Mul):
        return e
    c_part, nc_part = e.args_cnc()
    n_nc = len(nc_part)
    if n_nc == 0 or n_nc == 1:
        return e
    elif e.has(TensorProduct):
        current = nc_part[0]
        if not isinstance(current, TensorProduct):
            raise TypeError('TensorProduct expected, got: %r' % current)
        n_terms = len(current.args)
        new_args = list(current.args)
        for next in nc_part[1:]:
            # TODO: check the hilbert spaces of next and current here.
            if isinstance(next, TensorProduct):
                if n_terms != len(next.args):
                    raise QuantumError(
                        'TensorProducts of different lengths: %r and %r' %
                        (current, next)
                    )
                for i in range(len(new_args)):
                    new_args[i] = new_args[i] * next.args[i]
            else:
                # this won't quite work as we don't want next in the
                # TensorProduct
                for i in range(len(new_args)):
                    new_args[i] = new_args[i] * next
            current = next
        return Mul(*c_part) * TensorProduct(*new_args)
    else:
        return e


def tensor_product_simp(e, **hints):
    """Try to simplify and combine TensorProducts.

    In general this will try to pull expressions inside of ``TensorProducts``.
    It currently only works for relatively simple cases where the products have
    only scalars, raw ``TensorProducts``, not ``Add``, ``Pow``, ``Commutators``
    of ``TensorProducts``. It is best to see what it does by showing examples.

    Examples
    ========

    >>> from sympy.physics.quantum import tensor_product_simp
    >>> from sympy.physics.quantum import TensorProduct
    >>> from sympy import Symbol
    >>> A = Symbol('A',commutative=False)
    >>> B = Symbol('B',commutative=False)
    >>> C = Symbol('C',commutative=False)
    >>> D = Symbol('D',commutative=False)

    First see what happens to products of tensor products:

    >>> e = TensorProduct(A,B)*TensorProduct(C,D)
    >>> e
    AxB*CxD
    >>> tensor_product_simp(e)
    (A*C)x(B*D)

    This is the core logic of this function, and it works inside, powers, sums,
    commutators and anticommutators as well:

    >>> tensor_product_simp(e**2)
    (A*C)x(B*D)**2

    """
    if isinstance(e, Add):
        return Add(*[tensor_product_simp(arg) for arg in e.args])
    elif isinstance(e, Pow):
        return tensor_product_simp(e.base) ** e.exp
    elif isinstance(e, Mul):
        return tensor_product_simp_Mul(e)
    elif isinstance(e, Commutator):
        return Commutator(*[tensor_product_simp(arg) for arg in e.args])
