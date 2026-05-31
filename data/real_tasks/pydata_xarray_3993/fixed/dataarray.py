                         "us", "ns", "ps", "fs", "as"} or None, optional
            Unit to compute gradient. Only valid for datetime coordinate.

        Returns
        -------
        differentiated: DataArray

        See also
        --------
        numpy.gradient: corresponding numpy function

        Examples
        --------

        >>> da = xr.DataArray(
        ...     np.arange(12).reshape(4, 3),
        ...     dims=["x", "y"],
        ...     coords={"x": [0, 0.1, 1.1, 1.2]},
        ... )
        >>> da
        <xarray.DataArray (x: 4, y: 3)>
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        >>>
        >>> da.differentiate("x")
        <xarray.DataArray (x: 4, y: 3)>
        array([[30.        , 30.        , 30.        ],
               [27.54545455, 27.54545455, 27.54545455],
               [27.54545455, 27.54545455, 27.54545455],
               [30.        , 30.        , 30.        ]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        """
        ds = self._to_temp_dataset().differentiate(coord, edge_order, datetime_unit)
        return self._from_temp_dataset(ds)

    def integrate(
        self,
        coord: Union[Hashable, Sequence[Hashable]] = None,
        datetime_unit: str = None,
        *,
        dim: Union[Hashable, Sequence[Hashable]] = None,
    ) -> "DataArray":
        """Integrate along the given coordinate using the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord: hashable, or a sequence of hashable
            Coordinate(s) used for the integration.
        dim : hashable, or sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit: {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', \
                        'ps', 'fs', 'as'}, optional

        Returns
        -------
        integrated: DataArray

        See also
        --------
        numpy.trapz: corresponding numpy function

        Examples
        --------

        >>> da = xr.DataArray(
        ...     np.arange(12).reshape(4, 3),
        ...     dims=["x", "y"],
        ...     coords={"x": [0, 0.1, 1.1, 1.2]},
        ... )
        >>> da
        <xarray.DataArray (x: 4, y: 3)>
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        >>>
        >>> da.integrate("x")
        <xarray.DataArray (y: 3)>
        array([5.4, 6.6, 7.8])
        Dimensions without coordinates: y
        """
        ds = self._to_temp_dataset().integrate(dim, datetime_unit)
        return self._from_temp_dataset(ds)

    def unify_chunks(self) -> "DataArray":
        """Unify chunk size along all chunked dimensions of this DataArray.

        Returns
        -------

        DataArray with consistent chunk sizes for all dask-array variables
