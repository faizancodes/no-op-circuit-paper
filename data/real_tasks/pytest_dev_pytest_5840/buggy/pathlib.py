        return Path(input)
    else:
        return root.joinpath(input)


def fnmatch_ex(pattern, path):
    """FNMatcher port from py.path.common which works with PurePath() instances.

    The difference between this algorithm and PurePath.match() is that the latter matches "**" glob expressions
    for each part of the path, while this algorithm uses the whole path instead.

    For example:
        "tests/foo/bar/doc/test_foo.py" matches pattern "tests/**/doc/test*.py" with this algorithm, but not with
        PurePath.match().

    This algorithm was ported to keep backward-compatibility with existing settings which assume paths match according
    this logic.

    References:
    * https://bugs.python.org/issue29249
    * https://bugs.python.org/issue34731
    """
    path = PurePath(path)
    iswin32 = sys.platform.startswith("win")

    if iswin32 and sep not in pattern and posix_sep in pattern:
        # Running on Windows, the pattern has no Windows path separators,
        # and the pattern has one or more Posix path separators. Replace
        # the Posix path separators with the Windows path separator.
        pattern = pattern.replace(posix_sep, sep)

    if sep not in pattern:
        name = path.name
    else:
        name = str(path)
        if path.is_absolute() and not os.path.isabs(pattern):
            pattern = "*{}{}".format(os.sep, pattern)
    return fnmatch.fnmatch(name, pattern)


def parts(s):
    parts = s.split(sep)
    return {sep.join(parts[: i + 1]) or sep for i in range(len(parts))}


def unique_path(path):
    """Returns a unique path in case-insensitive (but case-preserving) file
    systems such as Windows.

    This is needed only for ``py.path.local``; ``pathlib.Path`` handles this
    natively with ``resolve()``."""
    return type(path)(normcase(str(path.realpath())))
