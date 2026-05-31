
        if weights.isnull().any():
            raise ValueError(
                "`weights` cannot contain missing values. "
                "Missing values can be replaced by `weights.fillna(0)`."
            )

        self.obj = obj
        self.weights = weights

    @staticmethod
    def _reduce(
        da: "DataArray",
        weights: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
        """reduce using dot; equivalent to (da * weights).sum(dim, skipna)

            for internal use only
        """

        # need to infer dims as we use `dot`
        if dim is None:
            dim = ...

        # need to mask invalid values in da, as `dot` does not implement skipna
        if skipna or (skipna is None and da.dtype.kind in "cfO"):
            da = da.fillna(0.0)

        # `dot` does not broadcast arrays, so this avoids creating a large
        # DataArray (if `weights` has additional dimensions)
        # maybe add fasttrack (`(da * weights).sum(dims=dim, skipna=skipna)`)
        return dot(da, weights, dims=dim)

    def _sum_of_weights(
        self, da: "DataArray", dim: Optional[Union[Hashable, Iterable[Hashable]]] = None
    ) -> "DataArray":
        """ Calculate the sum of weights, accounting for missing values """

        # we need to mask data values that are nan; else the weights are wrong
        mask = da.notnull()

        sum_of_weights = self._reduce(mask, self.weights, dim=dim, skipna=False)

        # 0-weights are not valid
        valid_weights = sum_of_weights != 0.0

        return sum_of_weights.where(valid_weights)

    def _weighted_sum(
        self,
        da: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
        """Reduce a DataArray by a by a weighted ``sum`` along some dimension(s)."""

        return self._reduce(da, self.weights, dim=dim, skipna=skipna)

    def _weighted_mean(
        self,
        da: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
        """Reduce a DataArray by a weighted ``mean`` along some dimension(s)."""

        weighted_sum = self._weighted_sum(da, dim=dim, skipna=skipna)

        sum_of_weights = self._sum_of_weights(da, dim=dim)

        return weighted_sum / sum_of_weights

    def _implementation(self, func, dim, **kwargs):

        raise NotImplementedError("Use `Dataset.weighted` or `DataArray.weighted`")

    def sum_of_weights(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        keep_attrs: Optional[bool] = None,
    ) -> Union["DataArray", "Dataset"]:

        return self._implementation(
            self._sum_of_weights, dim=dim, keep_attrs=keep_attrs
        )
