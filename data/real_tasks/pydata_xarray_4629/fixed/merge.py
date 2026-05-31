
def merge_data_and_coords(data, coords, compat="broadcast_equals", join="outer"):
    """Used in Dataset.__init__."""
    objects = [data, coords]
    explicit_coords = coords.keys()
    indexes = dict(_extract_indexes_from_coords(coords))
    return merge_core(
        objects, compat, join, explicit_coords=explicit_coords, indexes=indexes
    )


def _extract_indexes_from_coords(coords):
    """Yields the name & index of valid indexes from a mapping of coords"""
    for name, variable in coords.items():
        variable = as_variable(variable, name=name)
        if variable.dims == (name,):
            yield name, variable.to_index()


def assert_valid_explicit_coords(variables, dims, explicit_coords):
    """Validate explicit coordinate names/dims.

    Raise a MergeError if an explicit coord shares a name with a dimension
    but is comprised of arbitrary dimensions.
    """
    for coord_name in explicit_coords:
        if coord_name in dims and variables[coord_name].dims != (coord_name,):
            raise MergeError(
                "coordinate %s shares a name with a dataset dimension, but is "
                "not a 1D variable along that dimension. This is disallowed "
                "by the xarray data model." % coord_name
            )


def merge_attrs(variable_attrs, combine_attrs):
    """Combine attributes from different variables according to combine_attrs"""
    if not variable_attrs:
        # no attributes to merge
        return None

    if combine_attrs == "drop":
        return {}
    elif combine_attrs == "override":
        return dict(variable_attrs[0])
    elif combine_attrs == "no_conflicts":
        result = dict(variable_attrs[0])
        for attrs in variable_attrs[1:]:
            try:
                result = compat_dict_union(result, attrs)
            except ValueError:
                raise MergeError(
                    "combine_attrs='no_conflicts', but some values are not "
                    "the same. Merging %s with %s" % (str(result), str(attrs))
                )
        return result
    elif combine_attrs == "identical":
        result = dict(variable_attrs[0])
        for attrs in variable_attrs[1:]:
            if not dict_equiv(result, attrs):
                raise MergeError(
                    "combine_attrs='identical', but attrs differ. First is %s "
                    ", other is %s." % (str(result), str(attrs))
                )
        return result
    else:
        raise ValueError("Unrecognised value for combine_attrs=%s" % combine_attrs)


class _MergeResult(NamedTuple):
    variables: Dict[Hashable, Variable]
    coord_names: Set[Hashable]
    dims: Dict[Hashable, int]
    indexes: Dict[Hashable, pd.Index]
    attrs: Dict[Hashable, Any]


def merge_core(
    objects: Iterable["CoercibleMapping"],
    compat: str = "broadcast_equals",
    join: str = "outer",
    combine_attrs: Optional[str] = "override",
    priority_arg: Optional[int] = None,
    explicit_coords: Optional[Sequence] = None,
    indexes: Optional[Mapping[Hashable, pd.Index]] = None,
    fill_value: object = dtypes.NA,
) -> _MergeResult:
    """Core logic for merging labeled objects.
