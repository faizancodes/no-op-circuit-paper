    ==========

    simp : Boolean
        Let certain methods use trigsimp on their outputs

    """

    simp = False

    def __init__(self, inlist):
        """This is the constructor for the Vector class.  You shouldn't be
        calling this, it should only be used by other functions. You should be
        treating Vectors like you would with if you were doing the math by
        hand, and getting the first 3 from the standard basis vectors from a
        ReferenceFrame.

        The only exception is to create a zero vector:
        zv = Vector(0)

        """

        self.args = []
        if inlist == 0:
            inlist = []
        if isinstance(inlist, dict):
            d = inlist
        else:
            d = {}
            for inp in inlist:
                if inp[1] in d:
                    d[inp[1]] += inp[0]
                else:
                    d[inp[1]] = inp[0]

        for k, v in d.items():
            if v != Matrix([0, 0, 0]):
                self.args.append((v, k))

    def __hash__(self):
        return hash(tuple(self.args))

    def __add__(self, other):
        """The add operator for Vector. """
        if other == 0:
            return self
        other = _check_vector(other)
        return Vector(self.args + other.args)

    def __and__(self, other):
        """Dot product of two vectors.

        Returns a scalar, the dot product of the two Vectors

        Parameters
        ==========

        other : Vector
            The Vector which we are dotting with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dot
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> dot(N.x, N.x)
        1
        >>> dot(N.x, N.y)
        0
        >>> A = N.orientnew('A', 'Axis', [q1, N.x])
        >>> dot(N.y, A.y)
        cos(q1)

        """

        from sympy.physics.vector.dyadic import Dyadic
        if isinstance(other, Dyadic):
            return NotImplemented
        other = _check_vector(other)
        out = S(0)
        for i, v1 in enumerate(self.args):
            for j, v2 in enumerate(other.args):
                out += ((v2[0].T)
                        * (v2[1].dcm(v1[1]))
                        * (v1[0]))[0]
        if Vector.simp:
            return trigsimp(sympify(out), recursive=True)
