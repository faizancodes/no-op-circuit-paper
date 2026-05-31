    module : module or `str`
        An imported module of which to check the version, or the name of
        that module (in which case an import of that module is attempted--
        if this fails `False` is returned).

    version : `str`
        The version as a string that this module must have at a minimum (e.g.
        ``'0.12'``).

    inclusive : `bool`
        The specified version meets the requirement inclusively (i.e. ``>=``)
        as opposed to strictly greater than (default: `True`).

    version_path : `str`
        A dotted attribute path to follow in the module for the version.
        Defaults to just ``'__version__'``, which should work for most Python
        modules.

    Examples
    --------

    >>> import astropy
    >>> minversion(astropy, '0.4.4')
    True
    """
    if isinstance(module, types.ModuleType):
        module_name = module.__name__
    elif isinstance(module, str):
        module_name = module
        try:
            module = resolve_name(module_name)
        except ImportError:
            return False
    else:
        raise ValueError('module argument must be an actual imported '
                         'module, or the import name of the module; '
                         'got {0!r}'.format(module))

    if '.' not in version_path:
        have_version = getattr(module, version_path)
    else:
        have_version = resolve_name(module.__name__, version_path)

    if inclusive:
        return LooseVersion(have_version) >= LooseVersion(version)
    else:
        return LooseVersion(have_version) > LooseVersion(version)


def find_current_module(depth=1, finddiff=False):
    """
    Determines the module/package from which this function is called.

    This function has two modes, determined by the ``finddiff`` option. it
    will either simply go the requested number of frames up the call
    stack (if ``finddiff`` is False), or it will go up the call stack until
    it reaches a module that is *not* in a specified set.

    Parameters
    ----------
    depth : int
        Specifies how far back to go in the call stack (0-indexed, so that
        passing in 0 gives back `astropy.utils.misc`).
    finddiff : bool or list
        If False, the returned ``mod`` will just be ``depth`` frames up from
        the current frame. Otherwise, the function will start at a frame
        ``depth`` up from current, and continue up the call stack to the
        first module that is *different* from those in the provided list.
        In this case, ``finddiff`` can be a list of modules or modules
        names. Alternatively, it can be True, which will use the module
        ``depth`` call stack frames up as the module the returned module
        most be different from.

    Returns
    -------
    mod : module or None
        The module object or None if the package cannot be found. The name of
        the module is available as the ``__name__`` attribute of the returned
        object (if it isn't None).

    Raises
    ------
    ValueError
        If ``finddiff`` is a list with an invalid entry.

    Examples
