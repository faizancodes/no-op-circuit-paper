            return self.transformers

    @_transformers.setter
    def _transformers(self, value):
        try:
            self.transformers = [
                (name, trans, col)
                for ((name, trans), (_, _, col)) in zip(value, self.transformers)
            ]
        except (TypeError, ValueError):
            self.transformers = value

    def set_output(self, *, transform=None):
        """Set the output container when `"transform"` and `"fit_transform"` are called.

        Calling `set_output` will set the output of all estimators in `transformers`
        and `transformers_`.

        Parameters
        ----------
        transform : {"default", "pandas"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: DataFrame output
            - `None`: Transform configuration is unchanged

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        super().set_output(transform=transform)
        transformers = (
            trans
            for _, trans, _ in chain(
                self.transformers, getattr(self, "transformers_", [])
            )
            if trans not in {"passthrough", "drop"}
        )
        for trans in transformers:
            _safe_set_output(trans, transform=transform)

        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `transformers` of the
        `ColumnTransformer`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self._get_params("_transformers", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that you
        can directly set the parameters of the estimators contained in
        `transformers` of `ColumnTransformer`.

        Parameters
        ----------
        **kwargs : dict
            Estimator parameters.

        Returns
        -------
        self : ColumnTransformer
            This estimator.
        """
        self._set_params("_transformers", **kwargs)
        return self

    def _iter(self, fitted=False, replace_strings=False, column_as_strings=False):
