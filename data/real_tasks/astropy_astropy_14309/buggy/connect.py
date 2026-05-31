from .util import first

# Keywords to remove for all tables that are read in
REMOVE_KEYWORDS = [
    "XTENSION",
    "BITPIX",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "PCOUNT",
    "GCOUNT",
    "TFIELDS",
    "THEAP",
]

# Column-specific keywords regex
COLUMN_KEYWORD_REGEXP = "(" + "|".join(KEYWORD_NAMES) + ")[0-9]+"


def is_column_keyword(keyword):
    return re.match(COLUMN_KEYWORD_REGEXP, keyword) is not None


def is_fits(origin, filepath, fileobj, *args, **kwargs):
    """
    Determine whether `origin` is a FITS file.

    Parameters
    ----------
    origin : str or readable file-like
        Path or file object containing a potential FITS file.

    Returns
    -------
    is_fits : bool
        Returns `True` if the given file is a FITS file.
    """
    if fileobj is not None:
        pos = fileobj.tell()
        sig = fileobj.read(30)
        fileobj.seek(pos)
        return sig == FITS_SIGNATURE
    elif filepath is not None:
        if filepath.lower().endswith(
            (".fits", ".fits.gz", ".fit", ".fit.gz", ".fts", ".fts.gz")
        ):
            return True
    return isinstance(args[0], (HDUList, TableHDU, BinTableHDU, GroupsHDU))


def _decode_mixins(tbl):
    """Decode a Table ``tbl`` that has astropy Columns + appropriate meta-data into
    the corresponding table with mixin columns (as appropriate).
    """
    # If available read in __serialized_columns__ meta info which is stored
    # in FITS COMMENTS between two sentinels.
    try:
        i0 = tbl.meta["comments"].index("--BEGIN-ASTROPY-SERIALIZED-COLUMNS--")
        i1 = tbl.meta["comments"].index("--END-ASTROPY-SERIALIZED-COLUMNS--")
    except (ValueError, KeyError):
        return tbl

    # The YAML data are split into COMMENT cards, with lines longer than 70
    # characters being split with a continuation character \ (backslash).
    # Strip the backslashes and join together.
    continuation_line = False
    lines = []
    for line in tbl.meta["comments"][i0 + 1 : i1]:
        if continuation_line:
            lines[-1] = lines[-1] + line[:70]
        else:
            lines.append(line[:70])
        continuation_line = len(line) == 71

    del tbl.meta["comments"][i0 : i1 + 1]
    if not tbl.meta["comments"]:
        del tbl.meta["comments"]

    info = meta.get_header_from_yaml(lines)

    # Add serialized column information to table meta for use in constructing mixins
    tbl.meta["__serialized_columns__"] = info["meta"]["__serialized_columns__"]

    # Use the `datatype` attribute info to update column attributes that are
    # NOT already handled via standard FITS column keys (name, dtype, unit).
    for col in info["datatype"]:
        for attr in ["description", "meta"]:
            if attr in col:
                setattr(tbl[col["name"]].info, attr, col[attr])
