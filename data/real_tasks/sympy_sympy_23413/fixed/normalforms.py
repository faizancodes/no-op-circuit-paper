    it is critical that x, y not only satisfy the condition of being small
    in magnitude -- namely that |x| <= |b|/g, |y| <- |a|/g -- but also that
    y == 0 when a | b.

    """
    x, y, g = ZZ.gcdex(a, b)
    if a != 0 and b % a == 0:
        y = 0
        x = -1 if a < 0 else 1
    return x, y, g


def _hermite_normal_form(A):
    r"""
    Compute the Hermite Normal Form of DomainMatrix *A* over :ref:`ZZ`.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over domain :ref:`ZZ`.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.5.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    # We work one row at a time, starting from the bottom row, and working our
    # way up.
    m, n = A.shape
    A = A.to_dense().rep.copy()
    # Our goal is to put pivot entries in the rightmost columns.
    # Invariant: Before processing each row, k should be the index of the
    # leftmost column in which we have so far put a pivot.
    k = n
    for i in range(m - 1, -1, -1):
        if k == 0:
            # This case can arise when n < m and we've already found n pivots.
            # We don't need to consider any more rows, because this is already
            # the maximum possible number of pivots.
            break
        k -= 1
        # k now points to the column in which we want to put a pivot.
        # We want zeros in all entries to the left of the pivot column.
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                # Replace cols j, k by lin combs of these cols such that, in row i,
                # col j has 0, while col k has the gcd of their row i entries. Note
                # that this ensures a nonzero entry in col k.
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                add_columns(A, k, j, u, v, -s, r)
        b = A[i][k]
        # Do not want the pivot entry to be negative.
        if b < 0:
            add_columns(A, k, k, -1, 0, -1, 0)
            b = -b
        # The pivot entry will be 0 iff the row was 0 from the pivot col all the
        # way to the left. In this case, we are still working on the same pivot
        # col for the next row. Therefore:
        if b == 0:
            k += 1
        # If the pivot entry is nonzero, then we want to reduce all entries to its
        # right in the sense of the division algorithm, i.e. make them all remainders
        # w.r.t. the pivot as divisor.
        else:
            for j in range(k + 1, n):
                q = A[i][j] // b
                add_columns(A, j, k, 1, -q, 0, 1)
    # Finally, the HNF consists of those columns of A in which we succeeded in making
    # a nonzero pivot.
    return DomainMatrix.from_rep(A)[:, k:]


def _hermite_normal_form_modulo_D(A, D):
    r"""
    Perform the mod *D* Hermite Normal Form reduction algorithm on
    :py:class:`~.DomainMatrix` *A*.

    Explanation
    ===========

    If *A* is an $m \times n$ matrix of rank $m$, having Hermite Normal Form
    $W$, and if *D* is any positive integer known in advance to be a multiple
