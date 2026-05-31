    Returns
    -------
    out : same as object
        New object of ones with the same shape and type as other.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     np.arange(6).reshape(2, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": [1, 2], "lon": [0, 1, 2]},
    ... )
    >>> x
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
      * lat      (lat) int64 1 2
      * lon      (lon) int64 0 1 2

    >>> xr.ones_like(x)
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[1, 1, 1],
           [1, 1, 1]])
    Coordinates:
      * lat      (lat) int64 1 2
      * lon      (lon) int64 0 1 2

    See Also
    --------
    zeros_like
    full_like

    """
    return full_like(other, 1, dtype)


def get_chunksizes(
    variables: Iterable[Variable],
) -> Mapping[Any, tuple[int, ...]]:

    chunks: dict[Any, tuple[int, ...]] = {}
    for v in variables:
        if hasattr(v.data, "chunks"):
            for dim, c in v.chunksizes.items():
                if dim in chunks and c != chunks[dim]:
                    raise ValueError(
                        f"Object has inconsistent chunks along dimension {dim}. "
                        "This can be fixed by calling unify_chunks()."
                    )
                chunks[dim] = c
    return Frozen(chunks)


def is_np_datetime_like(dtype: DTypeLike) -> bool:
    """Check if a dtype is a subclass of the numpy datetime types"""
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)


def is_np_timedelta_like(dtype: DTypeLike) -> bool:
    """Check whether dtype is of the timedelta64 dtype."""
    return np.issubdtype(dtype, np.timedelta64)


def _contains_cftime_datetimes(array) -> bool:
    """Check if an array contains cftime.datetime objects"""
    if cftime is None:
        return False
    else:
        if array.dtype == np.dtype("O") and array.size > 0:
            sample = array.ravel()[0]
            if is_duck_dask_array(sample):
                sample = sample.compute()
                if isinstance(sample, np.ndarray):
                    sample = sample.item()
            return isinstance(sample, cftime.datetime)
        else:
            return False


def contains_cftime_datetimes(var) -> bool:
    """Check if an xarray.Variable contains cftime.datetime objects"""
    if var.dtype == np.dtype("O") and var.size > 0:
        return _contains_cftime_datetimes(var.data)
    else:
        return False
