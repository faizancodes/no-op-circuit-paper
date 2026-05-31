        Parameters
        ----------
        data : object (column-like sequence)
            Input column data
        copy : bool
            Make a copy
        default_name : str
            Default name
        dtype : np.dtype or None
            Data dtype
        name : str or None
            Column name

        Returns
        -------
        col : Column, MaskedColumn, mixin-column type
            Object that can be used as a column in self
        """

        data_is_mixin = self._is_mixin_for_table(data)
        masked_col_cls = (self.ColumnClass
                          if issubclass(self.ColumnClass, self.MaskedColumn)
                          else self.MaskedColumn)

        try:
            data0_is_mixin = self._is_mixin_for_table(data[0])
        except Exception:
            # Need broad exception, cannot predict what data[0] raises for arbitrary data
            data0_is_mixin = False

        # If the data is not an instance of Column or a mixin class, we can
        # check the registry of mixin 'handlers' to see if the column can be
        # converted to a mixin class
        if (handler := get_mixin_handler(data)) is not None:
            original_data = data
            data = handler(data)
            if not (data_is_mixin := self._is_mixin_for_table(data)):
                fully_qualified_name = (original_data.__class__.__module__ + '.'
                                        + original_data.__class__.__name__)
                raise TypeError('Mixin handler for object of type '
                                f'{fully_qualified_name} '
                                'did not return a valid mixin column')

        # Structured ndarray gets viewed as a mixin unless already a valid
        # mixin class
        if (not isinstance(data, Column) and not data_is_mixin
                and isinstance(data, np.ndarray) and len(data.dtype) > 1):
            data = data.view(NdarrayMixin)
            data_is_mixin = True

        # Get the final column name using precedence.  Some objects may not
        # have an info attribute. Also avoid creating info as a side effect.
        if not name:
            if isinstance(data, Column):
                name = data.name or default_name
            elif 'info' in getattr(data, '__dict__', ()):
                name = data.info.name or default_name
            else:
                name = default_name

        if isinstance(data, Column):
            # If self.ColumnClass is a subclass of col, then "upgrade" to ColumnClass,
            # otherwise just use the original class.  The most common case is a
            # table with masked=True and ColumnClass=MaskedColumn.  Then a Column
            # gets upgraded to MaskedColumn, but the converse (pre-4.0) behavior
            # of downgrading from MaskedColumn to Column (for non-masked table)
            # does not happen.
            col_cls = self._get_col_cls_for_table(data)

        elif data_is_mixin:
            # Copy the mixin column attributes if they exist since the copy below
            # may not get this attribute.
            col = col_copy(data, copy_indices=self._init_indices) if copy else data
            col.info.name = name
            return col

        elif data0_is_mixin:
            # Handle case of a sequence of a mixin, e.g. [1*u.m, 2*u.m].
            try:
                col = data[0].__class__(data)
                col.info.name = name
                return col
            except Exception:
                # If that didn't work for some reason, just turn it into np.array of object
                data = np.array(data, dtype=object)
                col_cls = self.ColumnClass

        elif isinstance(data, (np.ma.MaskedArray, Masked)):
            # Require that col_cls be a subclass of MaskedColumn, remembering
            # that ColumnClass could be a user-defined subclass (though more-likely
            # could be MaskedColumn).
            col_cls = masked_col_cls
