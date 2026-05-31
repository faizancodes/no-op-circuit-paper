    <xarray.Dataset>
    Dimensions:         (x: 3)
    Coords:
      * position        (x) int64   0 1 2
    Data variables:
        temperature     (x) float64 11.04 23.57 20.77 ...

    >>> x2
    <xarray.Dataset>
    Dimensions:         (x: 3)
    Coords:
      * position        (x) int64   3 4 5
    Data variables:
        temperature     (x) float64 6.97 8.13 7.42 ...

    >>> combined = xr.combine_by_coords([x2, x1])
    <xarray.Dataset>
    Dimensions:         (x: 6)
    Coords:
      * position        (x) int64   0 1 2 3 4 5
    Data variables:
        temperature     (x) float64 11.04 23.57 20.77 ...
    """

    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    # Perform the multidimensional combine on each group of data variables
    # before merging back together
    concatenated_grouped_by_data_vars = []
    for vars, datasets_with_same_vars in grouped_by_vars:
        combined_ids, concat_dims = _infer_concat_order_from_coords(
            list(datasets_with_same_vars))

        _check_shape_tile_ids(combined_ids)

        # Concatenate along all of concat_dims one by one to create single ds
        concatenated = _combine_nd(combined_ids, concat_dims=concat_dims,
                                   data_vars=data_vars, coords=coords,
                                   fill_value=fill_value)

        # Check the overall coordinates are monotonically increasing
        for dim in concatenated.dims:
            if dim in concatenated:
                indexes = concatenated.indexes.get(dim)
                if not (indexes.is_monotonic_increasing
                        or indexes.is_monotonic_decreasing):
                    raise ValueError("Resulting object does not have monotonic"
                                     " global indexes along dimension {}"
                                     .format(dim))
        concatenated_grouped_by_data_vars.append(concatenated)

    return merge(concatenated_grouped_by_data_vars, compat=compat,
                 fill_value=fill_value)


# Everything beyond here is only needed until the deprecation cycle in #2616
# is completed


_CONCAT_DIM_DEFAULT = '__infer_concat_dim__'


def auto_combine(datasets, concat_dim='_not_supplied', compat='no_conflicts',
                 data_vars='all', coords='different', fill_value=dtypes.NA,
                 from_openmfds=False):
    """
    Attempt to auto-magically combine the given datasets into one.

    This entire function is deprecated in favour of ``combine_nested`` and
    ``combine_by_coords``.

    This method attempts to combine a list of datasets into a single entity by
    inspecting metadata and using a combination of concat and merge.
    It does not concatenate along more than one dimension or sort data under
    any circumstances. It does align coordinates, but different variables on
    datasets can cause it to fail under some scenarios. In complex cases, you
    may need to clean up your data and use ``concat``/``merge`` explicitly.
    ``auto_combine`` works well if you have N years of data and M data
    variables, and each combination of a distinct time period and set of data
    variables is saved its own dataset.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Dataset objects to merge.
    concat_dim : str or DataArray or Index, optional
        Dimension along which to concatenate variables, as used by
        :py:func:`xarray.concat`. You only need to provide this argument if
        the dimension along which you want to concatenate is not a dimension
        in the original datasets, e.g., if you want to stack a collection of
        2D arrays along a third dimension.
        By default, xarray attempts to infer this argument by examining
