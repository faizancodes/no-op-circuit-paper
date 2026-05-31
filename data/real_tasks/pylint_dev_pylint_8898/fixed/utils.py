    """Return a list of stripped string by splitting the string given as
    argument on `sep` (',' by default), empty strings are discarded.

    >>> _splitstrip('a, b, c   ,  4,,')
    ['a', 'b', 'c', '4']
    >>> _splitstrip('a')
    ['a']
    >>> _splitstrip('a,\nb,\nc,')
    ['a', 'b', 'c']

    :type string: str or unicode
    :param string: a csv line

    :type sep: str or unicode
    :param sep: field separator, default to the comma (',')

    :rtype: str or unicode
    :return: the unquoted string (or the input string if it wasn't quoted)
    """
    return [word.strip() for word in string.split(sep) if word.strip()]


def _unquote(string: str) -> str:
    """Remove optional quotes (simple or double) from the string.

    :param string: an optionally quoted string
    :return: the unquoted string (or the input string if it wasn't quoted)
    """
    if not string:
        return string
    if string[0] in "\"'":
        string = string[1:]
    if string[-1] in "\"'":
        string = string[:-1]
    return string


def _check_csv(value: list[str] | tuple[str] | str) -> Sequence[str]:
    if isinstance(value, (list, tuple)):
        return value
    return _splitstrip(value)


def _check_regexp_csv(value: list[str] | tuple[str] | str) -> Iterable[str]:
    r"""Split a comma-separated list of regexps, taking care to avoid splitting
    a regex employing a comma as quantifier, as in `\d{1,2}`."""
    if isinstance(value, (list, tuple)):
        yield from value
    else:
        # None is a sentinel value here
        regexps: deque[deque[str] | None] = deque([None])
        open_braces = False
        for char in value:
            if char == "{":
                open_braces = True
            elif char == "}" and open_braces:
                open_braces = False

            if char == "," and not open_braces:
                regexps.append(None)
            elif regexps[-1] is None:
                regexps.pop()
                regexps.append(deque([char]))
            else:
                regexps[-1].append(char)
        yield from ("".join(regexp).strip() for regexp in regexps if regexp is not None)


def _comment(string: str) -> str:
    """Return string as a comment."""
    lines = [line.strip() for line in string.splitlines()]
    sep = "\n"
    return "# " + f"{sep}# ".join(lines)


def _format_option_value(optdict: OptionDict, value: Any) -> str:
    """Return the user input's value from a 'compiled' value.

    TODO: Refactor the code to not use this deprecated function
    """
    if optdict.get("type", None) == "py_version":
        value = ".".join(str(item) for item in value)
    elif isinstance(value, (list, tuple)):
        value = ",".join(_format_option_value(optdict, item) for item in value)
    elif isinstance(value, dict):
        value = ",".join(f"{k}:{v}" for k, v in value.items())
    elif hasattr(value, "match"):  # optdict.get('type') == 'regexp'
        # compiled regexp
        value = value.pattern
    elif optdict.get("type") == "yn":
        value = "yes" if value else "no"
    elif isinstance(value, str) and value.isspace():
        value = f"'{value}'"
    return str(value)


def format_section(
    stream: TextIO,
    section: str,
    options: list[tuple[str, OptionDict, Any]],
    doc: str | None = None,
) -> None:
    """Format an option's section using the INI format."""
    warnings.warn(
        "format_section has been deprecated. It will be removed in pylint 3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    if doc:
        print(_comment(doc), file=stream)
    print(f"[{section}]", file=stream)
