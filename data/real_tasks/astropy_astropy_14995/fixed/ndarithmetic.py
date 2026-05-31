            axis_kwarg = dict(axis=kwds["axis"]) if "axis" in kwds else dict()
            return self.uncertainty.propagate(
                operation, operand, result, correlation, **axis_kwarg
            )

    def _arithmetic_mask(self, operation, operand, handle_mask, axis=None, **kwds):
        """
        Calculate the resulting mask.

        This is implemented as the piecewise ``or`` operation if both have a
        mask.

        Parameters
        ----------
        operation : callable
            see :meth:`NDArithmeticMixin._arithmetic` parameter description.
            By default, the ``operation`` will be ignored.

        operand : `NDData`-like instance
            The second operand wrapped in an instance of the same class as
            self.

        handle_mask : callable
            see :meth:`NDArithmeticMixin.add`

        kwds :
            Additional parameters given to ``handle_mask``.

        Returns
        -------
        result_mask : any type
            If only one mask was present this mask is returned.
            If neither had a mask ``None`` is returned. Otherwise
            ``handle_mask`` must create (and copy) the returned mask.
        """
        # If only one mask is present we need not bother about any type checks
        if (
            self.mask is None and operand is not None and operand.mask is None
        ) or handle_mask is None:
            return None
        elif self.mask is None and operand is not None:
            # Make a copy so there is no reference in the result.
            return deepcopy(operand.mask)
        elif operand.mask is None:
            return deepcopy(self.mask)
        else:
            # Now let's calculate the resulting mask (operation enforces copy)
            return handle_mask(self.mask, operand.mask, **kwds)

    def _arithmetic_wcs(self, operation, operand, compare_wcs, **kwds):
        """
        Calculate the resulting wcs.

        There is actually no calculation involved but it is a good place to
        compare wcs information of both operands. This is currently not working
        properly with `~astropy.wcs.WCS` (which is the suggested class for
        storing as wcs property) but it will not break it neither.

        Parameters
        ----------
        operation : callable
            see :meth:`NDArithmeticMixin._arithmetic` parameter description.
            By default, the ``operation`` will be ignored.

        operand : `NDData` instance or subclass
            The second operand wrapped in an instance of the same class as
            self.

        compare_wcs : callable
            see :meth:`NDArithmeticMixin.add` parameter description.

        kwds :
            Additional parameters given to ``compare_wcs``.

        Raises
        ------
        ValueError
            If ``compare_wcs`` returns ``False``.

        Returns
        -------
        result_wcs : any type
            The ``wcs`` of the first operand is returned.
        """
        # ok, not really arithmetic but we need to check which wcs makes sense
        # for the result and this is an ideal place to compare the two WCS,
        # too.

        # I'll assume that the comparison returned None or False in case they
        # are not equal.
