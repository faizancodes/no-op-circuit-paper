                        )
                    elif not isinstance(item, bytes):
                        item = str(item)
                    query_val.append(item)
        query_params.append((key, query_val))
    return original_urlencode(query_params, doseq)


def http_date(epoch_seconds=None):
    """
    Format the time to match the RFC1123 date format as specified by HTTP
    RFC7231 section 7.1.1.1.

    `epoch_seconds` is a floating point number expressed in seconds since the
    epoch, in UTC - such as that outputted by time.time(). If set to None, it
    defaults to the current time.

    Output a string in the format 'Wdy, DD Mon YYYY HH:MM:SS GMT'.
    """
    return formatdate(epoch_seconds, usegmt=True)


def parse_http_date(date):
    """
    Parse a date format as specified by HTTP RFC7231 section 7.1.1.1.

    The three formats allowed by the RFC are accepted, even if only the first
    one is still in widespread use.

    Return an integer expressed in seconds since the epoch, in UTC.
    """
    # email.utils.parsedate() does the job for RFC1123 dates; unfortunately
    # RFC7231 makes it mandatory to support RFC850 dates too. So we roll
    # our own RFC-compliant parsing.
    for regex in RFC1123_DATE, RFC850_DATE, ASCTIME_DATE:
        m = regex.match(date)
        if m is not None:
            break
    else:
        raise ValueError("%r is not in a valid HTTP date format" % date)
    try:
        year = int(m.group('year'))
        if year < 100:
            current_year = datetime.datetime.utcnow().year
            current_century = current_year - (current_year % 100)
            if year - (current_year % 100) > 50:
                # year that appears to be more than 50 years in the future are
                # interpreted as representing the past.
                year += current_century - 100
            else:
                year += current_century
        month = MONTHS.index(m.group('mon').lower()) + 1
        day = int(m.group('day'))
        hour = int(m.group('hour'))
        min = int(m.group('min'))
        sec = int(m.group('sec'))
        result = datetime.datetime(year, month, day, hour, min, sec)
        return calendar.timegm(result.utctimetuple())
    except Exception as exc:
        raise ValueError("%r is not a valid date" % date) from exc


def parse_http_date_safe(date):
    """
    Same as parse_http_date, but return None if the input is invalid.
    """
    try:
        return parse_http_date(date)
    except Exception:
        pass


# Base 36 functions: useful for generating compact URLs

def base36_to_int(s):
    """
    Convert a base 36 string to an int. Raise ValueError if the input won't fit
    into an int.
    """
    # To prevent overconsumption of server resources, reject any
    # base36 string that is longer than 13 base36 digits (13 digits
    # is sufficient to base36-encode any 64-bit integer)
    if len(s) > 13:
        raise ValueError("Base36 input too large")
    return int(s, 36)


def int_to_base36(i):
    """Convert an integer to a base36 string."""
    char_set = '0123456789abcdefghijklmnopqrstuvwxyz'
    if i < 0:
        raise ValueError("Negative base36 conversion input.")
    if i < 36:
        return char_set[i]
