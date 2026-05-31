                newD[i, j] = self._smat[i, j]
        self._smat = newD
        self.cols -= 1

    def col_join(self, other):
        """Returns B augmented beneath A (row-wise joining)::

            [A]
            [B]

        Examples
        ========

        >>> from sympy import SparseMatrix, Matrix, ones
        >>> A = SparseMatrix(ones(3))
        >>> A
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
        >>> B = SparseMatrix.eye(3)
        >>> B
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        >>> C = A.col_join(B); C
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
        >>> C == A.col_join(Matrix(B))
        True

        Joining along columns is the same as appending rows at the end
        of the matrix:

        >>> C == A.row_insert(A.rows, Matrix(B))
        True
        """
        if not self:
            return type(self)(other)
        A, B = self, other
        if not A.cols == B.cols:
            raise ShapeError()
        A = A.copy()
        if not isinstance(B, SparseMatrix):
            k = 0
            b = B._mat
            for i in range(B.rows):
                for j in range(B.cols):
                    v = b[k]
                    if v:
                        A._smat[(i + A.rows, j)] = v
                    k += 1
        else:
            for (i, j), v in B._smat.items():
                A._smat[i + A.rows, j] = v
        A.rows += B.rows
        return A

    def col_op(self, j, f):
        """In-place operation on col j using two-arg functor whose args are
        interpreted as (self[i, j], i) for i in range(self.rows).

        Examples
        ========

        >>> from sympy.matrices import SparseMatrix
        >>> M = SparseMatrix.eye(3)*2
        >>> M[1, 0] = -1
        >>> M.col_op(1, lambda v, i: v + 2*M[i, 0]); M
        Matrix([
        [ 2, 4, 0],
        [-1, 0, 0],
        [ 0, 0, 2]])
        """
        for i in range(self.rows):
            v = self._smat.get((i, j), S.Zero)
            fv = f(v, i)
            if fv:
                self._smat[(i, j)] = fv
            elif v:
                self._smat.pop((i, j))
