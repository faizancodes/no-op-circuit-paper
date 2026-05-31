        '''
        def _number_blocks(blocks):
            # number the blocks of a block system
            # in order and return the number of
            # blocks and the tuple with the
            # reordering
            n = len(blocks)
            appeared = {}
            m = 0
            b = [None]*n
            for i in range(n):
                if blocks[i] not in appeared:
                    appeared[blocks[i]] = m
                    b[i] = m
                    m += 1
                else:
                    b[i] = appeared[blocks[i]]
            return tuple(b), m

        if not self.is_transitive():
            return False
        blocks = []
        num_blocks = []
        rep_blocks = []
        if randomized:
            random_stab_gens = []
            v = self.schreier_vector(0)
            for i in range(len(self)):
                random_stab_gens.append(self.random_stab(0, v))
            stab = PermutationGroup(random_stab_gens)
        else:
            stab = self.stabilizer(0)
        orbits = stab.orbits()
        for orb in orbits:
            x = orb.pop()
            if x != 0:
                block = self.minimal_block([0, x])
                num_block, m = _number_blocks(block)
                # a representative block (containing 0)
                rep = {j for j in range(self.degree) if num_block[j] == 0}
                # check if the system is minimal with
                # respect to the already discovere ones
                minimal = True
                to_remove = []
                for i, r in enumerate(rep_blocks):
                    if len(r) > len(rep) and rep.issubset(r):
                        # i-th block system is not minimal
                        del num_blocks[i], blocks[i]
                        to_remove.append(rep_blocks[i])
                    elif len(r) < len(rep) and r.issubset(rep):
                        # the system being checked is not minimal
                        minimal = False
                        break
                # remove non-minimal representative blocks
                rep_blocks = [r for r in rep_blocks if r not in to_remove]

                if minimal and num_block not in num_blocks:
                    blocks.append(block)
                    num_blocks.append(num_block)
                    rep_blocks.append(rep)
        return blocks

    @property
    def is_solvable(self):
        """Test if the group is solvable.

        ``G`` is solvable if its derived series terminates with the trivial
        group ([1], p.29).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(3)
        >>> S.is_solvable
        True

        See Also
        ========

        is_nilpotent, derived_series

        """
        if self._is_solvable is None:
            if self.order() % 2 != 0:
                return True
            ds = self.derived_series()
            terminator = ds[len(ds) - 1]
            gens = terminator.generators
            degree = self.degree
            identity = _af_new(list(range(degree)))
            if all(g == identity for g in gens):
                self._is_solvable = True
                return True
            else:
                self._is_solvable = False
                return False
        else:
