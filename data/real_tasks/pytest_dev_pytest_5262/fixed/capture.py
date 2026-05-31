    """ return an open text file object that's a duplicate of f on the
        FD-level if possible.
    """
    encoding = getattr(f, "encoding", None)
    try:
        fd = f.fileno()
    except Exception:
        if "b" not in getattr(f, "mode", "") and hasattr(f, "encoding"):
            # we seem to have a text stream, let's just use it
            return f
    else:
        newfd = os.dup(fd)
        if "b" not in mode:
            mode += "b"
        f = os.fdopen(newfd, mode, 0)  # no buffering
    return EncodedFile(f, encoding or default_encoding)


class EncodedFile(object):
    errors = "strict"  # possibly needed by py3 code (issue555)

    def __init__(self, buffer, encoding):
        self.buffer = buffer
        self.encoding = encoding

    def write(self, obj):
        if isinstance(obj, six.text_type):
            obj = obj.encode(self.encoding, "replace")
        elif _PY3:
            raise TypeError(
                "write() argument must be str, not {}".format(type(obj).__name__)
            )
        self.buffer.write(obj)

    def writelines(self, linelist):
        data = "".join(linelist)
        self.write(data)

    @property
    def name(self):
        """Ensure that file.name is a string."""
        return repr(self.buffer)

    @property
    def mode(self):
        return self.buffer.mode.replace("b", "")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "buffer"), name)


CaptureResult = collections.namedtuple("CaptureResult", ["out", "err"])


class MultiCapture(object):
    out = err = in_ = None
    _state = None

    def __init__(self, out=True, err=True, in_=True, Capture=None):
        if in_:
            self.in_ = Capture(0)
        if out:
            self.out = Capture(1)
        if err:
            self.err = Capture(2)

    def __repr__(self):
        return "<MultiCapture out=%r err=%r in_=%r _state=%r _in_suspended=%r>" % (
            self.out,
            self.err,
            self.in_,
            self._state,
            getattr(self, "_in_suspended", "<UNSET>"),
        )

    def start_capturing(self):
        self._state = "started"
        if self.in_:
            self.in_.start()
        if self.out:
            self.out.start()
        if self.err:
            self.err.start()

    def pop_outerr_to_orig(self):
        """ pop current snapshot out/err capture and flush to orig streams. """
        out, err = self.readouterr()
        if out:
            self.out.writeorg(out)
        if err:
