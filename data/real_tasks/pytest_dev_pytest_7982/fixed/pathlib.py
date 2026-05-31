        if module_file.endswith((".pyc", ".pyo")):
            module_file = module_file[:-1]
        if module_file.endswith(os.path.sep + "__init__.py"):
            module_file = module_file[: -(len(os.path.sep + "__init__.py"))]

        try:
            is_same = os.path.samefile(str(path), module_file)
        except FileNotFoundError:
            is_same = False

        if not is_same:
            raise ImportPathMismatchError(module_name, module_file, path)

    return mod


def resolve_package_path(path: Path) -> Optional[Path]:
    """Return the Python package path by looking for the last
    directory upwards which still contains an __init__.py.

    Returns None if it can not be determined.
    """
    result = None
    for parent in itertools.chain((path,), path.parents):
        if parent.is_dir():
            if not parent.joinpath("__init__.py").is_file():
                break
            if not parent.name.isidentifier():
                break
            result = parent
    return result


def visit(
    path: str, recurse: Callable[["os.DirEntry[str]"], bool]
) -> Iterator["os.DirEntry[str]"]:
    """Walk a directory recursively, in breadth-first order.

    Entries at each directory level are sorted.
    """
    entries = sorted(os.scandir(path), key=lambda entry: entry.name)
    yield from entries
    for entry in entries:
        if entry.is_dir() and recurse(entry):
            yield from visit(entry.path, recurse)


def absolutepath(path: Union[Path, str]) -> Path:
    """Convert a path to an absolute path using os.path.abspath.

    Prefer this over Path.resolve() (see #6523).
    Prefer this over Path.absolute() (not public, doesn't normalize).
    """
    return Path(os.path.abspath(str(path)))


def commonpath(path1: Path, path2: Path) -> Optional[Path]:
    """Return the common part shared with the other path, or None if there is
    no common part.

    If one path is relative and one is absolute, returns None.
    """
    try:
        return Path(os.path.commonpath((str(path1), str(path2))))
    except ValueError:
        return None


def bestrelpath(directory: Path, dest: Path) -> str:
    """Return a string which is a relative path from directory to dest such
    that directory/bestrelpath == dest.

    The paths must be either both absolute or both relative.

    If no such path can be determined, returns dest.
    """
    if dest == directory:
        return os.curdir
    # Find the longest common directory.
    base = commonpath(directory, dest)
    # Can be the case on Windows for two absolute paths on different drives.
    # Can be the case for two relative paths without common prefix.
    # Can be the case for a relative path and an absolute path.
    if not base:
        return str(dest)
    reldirectory = directory.relative_to(base)
    reldest = dest.relative_to(base)
