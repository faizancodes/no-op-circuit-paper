    """
    Return a sequence of equally spaced Matplotlib dates.

    The dates start at *dstart* and reach up to, but not including *dend*.
    They are spaced by *delta*.

    Parameters
    ----------
    dstart, dend : `~datetime.datetime`
        The date limits.
    delta : `datetime.timedelta`
        Spacing of the dates.

    Returns
    -------
    `numpy.array`
        A list floats representing Matplotlib dates.

    """
    f1 = date2num(dstart)
    f2 = date2num(dend)
    step = delta.total_seconds() / SEC_PER_DAY

    # calculate the difference between dend and dstart in times of delta
    num = int(np.ceil((f2 - f1) / step))

    # calculate end of the interval which will be generated
    dinterval_end = dstart + num * delta

    # ensure, that an half open interval will be generated [dstart, dend)
    if dinterval_end >= dend:
        # if the endpoint is greater than dend, just subtract one delta
        dinterval_end -= delta
        num -= 1

    f2 = date2num(dinterval_end)  # new float-endpoint
    return np.linspace(f1, f2, num + 1)


def _wrap_in_tex(text):
    p = r'([a-zA-Z]+)'
    ret_text = re.sub(p, r'}$\1$\\mathdefault{', text)

    # Braces ensure dashes are not spaced like binary operators.
    ret_text = '$\\mathdefault{'+ret_text.replace('-', '{-}')+'}$'
    ret_text = ret_text.replace('$\\mathdefault{}$', '')
    return ret_text


## date tickers and formatters ###


class DateFormatter(ticker.Formatter):
    """
    Format a tick (in days since the epoch) with a
    `~datetime.datetime.strftime` format string.
    """

    def __init__(self, fmt, tz=None, *, usetex=None):
        """
        Parameters
        ----------
        fmt : str
            `~datetime.datetime.strftime` format string
        tz : `datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone.
        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            results of the formatter.
        """
        if tz is None:
            tz = _get_rc_timezone()
        self.fmt = fmt
        self.tz = tz
        self._usetex = (usetex if usetex is not None else
                        mpl.rcParams['text.usetex'])

    def __call__(self, x, pos=0):
        result = num2date(x, self.tz).strftime(self.fmt)
        return _wrap_in_tex(result) if self._usetex else result

    def set_tzinfo(self, tz):
        self.tz = tz


class ConciseDateFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use for the
