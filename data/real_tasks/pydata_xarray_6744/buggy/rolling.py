
    def __init__(
        self,
        obj: DataArray,
        windows: Mapping[Any, int],
        min_periods: int | None = None,
        center: bool | Mapping[Any, bool] = False,
    ) -> None:
        """
        Moving window object for DataArray.
        You should use DataArray.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool, default: False
            Set the labels at the center of the window.

        Returns
        -------
        rolling : type of input argument

        See Also
        --------
        xarray.DataArray.rolling
        xarray.DataArray.groupby
        xarray.Dataset.rolling
        xarray.Dataset.groupby
        """
        super().__init__(obj, windows, min_periods=min_periods, center=center)

        # TODO legacy attribute
        self.window_labels = self.obj[self.dim[0]]

    def __iter__(self) -> Iterator[tuple[RollingKey, DataArray]]:
        if self.ndim > 1:
            raise ValueError("__iter__ is only supported for 1d-rolling")
        stops = np.arange(1, len(self.window_labels) + 1)
        starts = stops - int(self.window[0])
        starts[: int(self.window[0])] = 0
        for (label, start, stop) in zip(self.window_labels, starts, stops):
            window = self.obj.isel({self.dim[0]: slice(start, stop)})

            counts = window.count(dim=self.dim[0])
            window = window.where(counts >= self.min_periods)

            yield (label, window)

    def construct(
        self,
        window_dim: Hashable | Mapping[Any, Hashable] | None = None,
        stride: int | Mapping[Any, int] = 1,
        fill_value: Any = dtypes.NA,
        keep_attrs: bool | None = None,
        **window_dim_kwargs: Hashable,
    ) -> DataArray:
        """
        Convert this rolling object to xr.DataArray,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim : Hashable or dict-like to Hashable, optional
            A mapping from dimension name to the new window dimension names.
        stride : int or mapping of int, default: 1
            Size of stride for the rolling window.
        fill_value : default: dtypes.NA
            Filling value to match the dimension size.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **window_dim_kwargs : Hashable, optional
            The keyword arguments form of ``window_dim`` {dim: new_name, ...}.

        Returns
        -------
        DataArray that is a view of the original array. The returned array is
        not writeable.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))

        >>> rolling = da.rolling(b=3)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
