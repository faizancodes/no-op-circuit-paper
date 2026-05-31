            return type(self)(result, dtype=self.dtype)
        else:
            return self._convert_scalar(result)

    def transpose(self, order) -> pd.Index:
        return self.array  # self.array should be always one-dimensional

    def __repr__(self) -> str:
        return f"{type(self).__name__}(array={self.array!r}, dtype={self.dtype!r})"

    def copy(self, deep: bool = True) -> PandasIndexingAdapter:
        # Not the same as just writing `self.array.copy(deep=deep)`, as
        # shallow copies of the underlying numpy.ndarrays become deep ones
        # upon pickling
        # >>> len(pickle.dumps((self.array, self.array)))
        # 4000281
        # >>> len(pickle.dumps((self.array, self.array.copy(deep=False))))
        # 8000341
        array = self.array.copy(deep=True) if deep else self.array
        return type(self)(array, self._dtype)


class PandasMultiIndexingAdapter(PandasIndexingAdapter):
    """Handles explicit indexing for a pandas.MultiIndex.

    This allows creating one instance for each multi-index level while
    preserving indexing efficiency (memoized + might reuse another instance with
    the same multi-index).

    """

    __slots__ = ("array", "_dtype", "level", "adapter")

    def __init__(
        self,
        array: pd.MultiIndex,
        dtype: DTypeLike = None,
        level: str | None = None,
    ):
        super().__init__(array, dtype)
        self.level = level

    def __array__(self, dtype: DTypeLike = None) -> np.ndarray:
        if self.level is not None:
            return self.array.get_level_values(self.level).values
        else:
            return super().__array__(dtype)

    def _convert_scalar(self, item):
        if isinstance(item, tuple) and self.level is not None:
            idx = tuple(self.array.names).index(self.level)
            item = item[idx]
        return super()._convert_scalar(item)

    def __getitem__(self, indexer):
        result = super().__getitem__(indexer)
        if isinstance(result, type(self)):
            result.level = self.level

        return result

    def __repr__(self) -> str:
        if self.level is None:
            return super().__repr__()
        else:
            props = (
                f"(array={self.array!r}, level={self.level!r}, dtype={self.dtype!r})"
            )
            return f"{type(self).__name__}{props}"

    def _get_array_subset(self) -> np.ndarray:
        # used to speed-up the repr for big multi-indexes
        threshold = max(100, OPTIONS["display_values_threshold"] + 2)
        if self.size > threshold:
            pos = threshold // 2
            indices = np.concatenate([np.arange(0, pos), np.arange(-pos, 0)])
            subset = self[OuterIndexer((indices,))]
        else:
            subset = self

        return np.asarray(subset)

    def _repr_inline_(self, max_width: int) -> str:
        from xarray.core.formatting import format_array_flat

        if self.level is None:
            return "MultiIndex"
        else:
