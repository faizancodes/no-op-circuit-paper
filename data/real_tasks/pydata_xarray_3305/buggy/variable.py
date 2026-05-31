                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result
            is a scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile and a quantile dimension
            is added to the return array. The other dimensions are the
             dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, Dataset.quantile,
        DataArray.quantile
        """
        if isinstance(self.data, dask_array_type):
            raise TypeError(
                "quantile does not work for arrays stored as dask "
                "arrays. Load the data via .compute() or .load() "
                "prior to calling this method."
            )

        q = np.asarray(q, dtype=np.float64)

        new_dims = list(self.dims)
        if dim is not None:
            axis = self.get_axis_num(dim)
            if utils.is_scalar(dim):
                new_dims.remove(dim)
            else:
                for d in dim:
                    new_dims.remove(d)
        else:
            axis = None
            new_dims = []

        # only add the quantile dimension if q is array like
        if q.ndim != 0:
            new_dims = ["quantile"] + new_dims

        qs = np.nanpercentile(
            self.data, q * 100.0, axis=axis, interpolation=interpolation
        )
        return Variable(new_dims, qs)

    def rank(self, dim, pct=False):
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within that
        set.  Ranks begin at 1, not 0. If `pct`, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.

        Returns
        -------
        ranked : Variable

        See Also
        --------
        Dataset.rank, DataArray.rank
        """
        import bottleneck as bn

        data = self.data

        if isinstance(data, dask_array_type):
            raise TypeError(
                "rank does not work for arrays stored as dask "
                "arrays. Load the data via .compute() or .load() "
                "prior to calling this method."
            )
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "rank is not implemented for {} objects.".format(type(data))
            )

        axis = self.get_axis_num(dim)
