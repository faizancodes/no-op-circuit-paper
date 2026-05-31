                new_dims = [replace_dims.get(d, d) for d in v.dims]
                variables[k] = v._replace(dims=new_dims)

        coord_names = self._coord_names - set(drop_variables) | set(new_variables)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes_
        )

    def reset_index(
        self: T_Dataset,
        dims_or_levels: Hashable | Sequence[Hashable],
        drop: bool = False,
    ) -> T_Dataset:
        """Reset the specified index(es) or multi-index level(s).

        Parameters
        ----------
        dims_or_levels : Hashable or Sequence of Hashable
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, default: False
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.set_index
        """
        if isinstance(dims_or_levels, str) or not isinstance(dims_or_levels, Sequence):
            dims_or_levels = [dims_or_levels]

        invalid_coords = set(dims_or_levels) - set(self._indexes)
        if invalid_coords:
            raise ValueError(
                f"{tuple(invalid_coords)} are not coordinates with an index"
            )

        drop_indexes: set[Hashable] = set()
        drop_variables: set[Hashable] = set()
        seen: set[Index] = set()
        new_indexes: dict[Hashable, Index] = {}
        new_variables: dict[Hashable, Variable] = {}

        def drop_or_convert(var_names):
            if drop:
                drop_variables.update(var_names)
            else:
                base_vars = {
                    k: self._variables[k].to_base_variable() for k in var_names
                }
                new_variables.update(base_vars)

        for name in dims_or_levels:
            index = self._indexes[name]

            if index in seen:
                continue
            seen.add(index)

            idx_var_names = set(self.xindexes.get_all_coords(name))
            drop_indexes.update(idx_var_names)

            if isinstance(index, PandasMultiIndex):
                # special case for pd.MultiIndex
                level_names = index.index.names
                keep_level_vars = {
                    k: self._variables[k]
                    for k in level_names
                    if k not in dims_or_levels
                }

                if index.dim not in dims_or_levels and keep_level_vars:
                    # do not drop the multi-index completely
                    # instead replace it by a new (multi-)index with dropped level(s)
                    idx = index.keep_levels(keep_level_vars)
                    idx_vars = idx.create_variables(keep_level_vars)
                    new_indexes.update({k: idx for k in idx_vars})
                    new_variables.update(idx_vars)
                    if not isinstance(idx, PandasMultiIndex):
                        # multi-index reduced to single index
                        # backward compatibility: unique level coordinate renamed to dimension
                        drop_variables.update(keep_level_vars)
                    drop_or_convert(
                        [k for k in level_names if k not in keep_level_vars]
                    )
                else:
                    # always drop the multi-index dimension variable
                    drop_variables.add(index.dim)
                    drop_or_convert(level_names)
            else:
                drop_or_convert(idx_var_names)

        indexes = {k: v for k, v in self._indexes.items() if k not in drop_indexes}
        indexes.update(new_indexes)

        variables = {
            k: v for k, v in self._variables.items() if k not in drop_variables
        }
        variables.update(new_variables)

        coord_names = set(new_variables) | self._coord_names

        return self._replace(variables, coord_names=coord_names, indexes=indexes)

    def reorder_levels(
        self: T_Dataset,
        dim_order: Mapping[Any, Sequence[int | Hashable]] | None = None,
        **dim_order_kwargs: Sequence[int | Hashable],
    ) -> T_Dataset:
        """Rearrange index levels using input order.

        Parameters
        ----------
        dim_order : dict-like of Hashable to Sequence of int or Hashable, optional
            Mapping from names matching dimensions and values given
            by lists representing new level orders. Every given dimension
            must have a multi-index.
        **dim_order_kwargs : Sequence of int or Hashable, optional
            The keyword arguments form of ``dim_order``.
            One of dim_order or dim_order_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced
            coordinates.
        """
        dim_order = either_dict_or_kwargs(dim_order, dim_order_kwargs, "reorder_levels")
        variables = self._variables.copy()
        indexes = dict(self._indexes)
        new_indexes: dict[Hashable, Index] = {}
        new_variables: dict[Hashable, IndexVariable] = {}

        for dim, order in dim_order.items():
