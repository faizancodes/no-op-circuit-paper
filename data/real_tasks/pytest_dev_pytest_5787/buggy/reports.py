
    def __init__(self, nodeid, outcome, longrepr, result, sections=(), **extra):
        self.nodeid = nodeid
        self.outcome = outcome
        self.longrepr = longrepr
        self.result = result or []
        self.sections = list(sections)
        self.__dict__.update(extra)

    @property
    def location(self):
        return (self.fspath, None, self.fspath)

    def __repr__(self):
        return "<CollectReport {!r} lenresult={} outcome={!r}>".format(
            self.nodeid, len(self.result), self.outcome
        )


class CollectErrorRepr(TerminalRepr):
    def __init__(self, msg):
        self.longrepr = msg

    def toterminal(self, out):
        out.line(self.longrepr, red=True)


def pytest_report_to_serializable(report):
    if isinstance(report, (TestReport, CollectReport)):
        data = report._to_json()
        data["_report_type"] = report.__class__.__name__
        return data


def pytest_report_from_serializable(data):
    if "_report_type" in data:
        if data["_report_type"] == "TestReport":
            return TestReport._from_json(data)
        elif data["_report_type"] == "CollectReport":
            return CollectReport._from_json(data)
        assert False, "Unknown report_type unserialize data: {}".format(
            data["_report_type"]
        )
