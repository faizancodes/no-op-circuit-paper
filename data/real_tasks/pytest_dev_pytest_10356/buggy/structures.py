        return self.mark.args

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Alias for mark.kwargs."""
        return self.mark.kwargs

    @property
    def markname(self) -> str:
        """:meta private:"""
        return self.name  # for backward-compat (2.4.1 had this attr)

    def with_args(self, *args: object, **kwargs: object) -> "MarkDecorator":
        """Return a MarkDecorator with extra arguments added.

        Unlike calling the MarkDecorator, with_args() can be used even
        if the sole argument is a callable/class.
        """
        mark = Mark(self.name, args, kwargs, _ispytest=True)
        return MarkDecorator(self.mark.combined_with(mark), _ispytest=True)

    # Type ignored because the overloads overlap with an incompatible
    # return type. Not much we can do about that. Thankfully mypy picks
    # the first match so it works out even if we break the rules.
    @overload
    def __call__(self, arg: Markable) -> Markable:  # type: ignore[misc]
        pass

    @overload
    def __call__(self, *args: object, **kwargs: object) -> "MarkDecorator":
        pass

    def __call__(self, *args: object, **kwargs: object):
        """Call the MarkDecorator."""
        if args and not kwargs:
            func = args[0]
            is_class = inspect.isclass(func)
            if len(args) == 1 and (istestfunc(func) or is_class):
                store_mark(func, self.mark)
                return func
        return self.with_args(*args, **kwargs)


def get_unpacked_marks(obj: object) -> Iterable[Mark]:
    """Obtain the unpacked marks that are stored on an object."""
    mark_list = getattr(obj, "pytestmark", [])
    if not isinstance(mark_list, list):
        mark_list = [mark_list]
    return normalize_mark_list(mark_list)


def normalize_mark_list(
    mark_list: Iterable[Union[Mark, MarkDecorator]]
) -> Iterable[Mark]:
    """
    Normalize an iterable of Mark or MarkDecorator objects into a list of marks
    by retrieving the `mark` attribute on MarkDecorator instances.

    :param mark_list: marks to normalize
    :returns: A new list of the extracted Mark objects
    """
    for mark in mark_list:
        mark_obj = getattr(mark, "mark", mark)
        if not isinstance(mark_obj, Mark):
            raise TypeError(f"got {repr(mark_obj)} instead of Mark")
        yield mark_obj


def store_mark(obj, mark: Mark) -> None:
    """Store a Mark on an object.

    This is used to implement the Mark declarations/decorators correctly.
    """
    assert isinstance(mark, Mark), mark
    # Always reassign name to avoid updating pytestmark in a reference that
    # was only borrowed.
    obj.pytestmark = [*get_unpacked_marks(obj), mark]


# Typing for builtin pytest marks. This is cheating; it gives builtin marks
# special privilege, and breaks modularity. But practicality beats purity...
if TYPE_CHECKING:
    from _pytest.scope import _ScopeName

    class _SkipMarkDecorator(MarkDecorator):
        @overload  # type: ignore[override,misc,no-overload-impl]
        def __call__(self, arg: Markable) -> Markable:
            ...

        @overload
        def __call__(self, reason: str = ...) -> "MarkDecorator":
            ...
