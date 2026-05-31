    except (ValueError, TypeError):
        try:
            return float(s)
        except (ValueError, TypeError) as e:
            raise ValueError(str(e))


def _format_value(value):
    """
    Converts a card value to its appropriate string representation as
    defined by the FITS format.
    """
    # string value should occupies at least 8 columns, unless it is
    # a null string
    if isinstance(value, str):
        if value == "":
            return "''"
        else:
            exp_val_str = value.replace("'", "''")
            val_str = f"'{exp_val_str:8}'"
            return f"{val_str:20}"

    # must be before int checking since bool is also int
    elif isinstance(value, (bool, np.bool_)):
        return f"{repr(value)[0]:>20}"  # T or F

    elif _is_int(value):
        return f"{value:>20d}"

    elif isinstance(value, (float, np.floating)):
        return f"{_format_float(value):>20}"

    elif isinstance(value, (complex, np.complexfloating)):
        val_str = f"({_format_float(value.real)}, {_format_float(value.imag)})"
        return f"{val_str:>20}"

    elif isinstance(value, Undefined):
        return ""
    else:
        return ""


def _format_float(value):
    """Format a floating number to make sure it gets the decimal point."""
    value_str = f"{value:.16G}"
    if "." not in value_str and "E" not in value_str:
        value_str += ".0"
    elif "E" in value_str:
        # On some Windows builds of Python (and possibly other platforms?) the
        # exponent is zero-padded out to, it seems, three digits.  Normalize
        # the format to pad only to two digits.
        significand, exponent = value_str.split("E")
        if exponent[0] in ("+", "-"):
            sign = exponent[0]
            exponent = exponent[1:]
        else:
            sign = ""
        value_str = f"{significand}E{sign}{int(exponent):02d}"

    # Limit the value string to at most 20 characters.
    str_len = len(value_str)

    if str_len > 20:
        idx = value_str.find("E")

        if idx < 0:
            value_str = value_str[:20]
        else:
            value_str = value_str[: 20 - (str_len - idx)] + value_str[idx:]

    return value_str


def _pad(input):
    """Pad blank space to the input string to be multiple of 80."""
    _len = len(input)
    if _len == Card.length:
        return input
    elif _len > Card.length:
        strlen = _len % Card.length
        if strlen == 0:
            return input
        else:
            return input + " " * (Card.length - strlen)

    # minimum length is 80
    else:
        strlen = _len % Card.length
        return input + " " * (Card.length - strlen)
