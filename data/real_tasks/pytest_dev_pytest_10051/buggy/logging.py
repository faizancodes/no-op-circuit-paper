class catching_logs:
    """Context manager that prepares the whole logging machinery properly."""

    __slots__ = ("handler", "level", "orig_level")

    def __init__(self, handler: _HandlerType, level: Optional[int] = None) -> None:
        self.handler = handler
        self.level = level

    def __enter__(self):
        root_logger = logging.getLogger()
        if self.level is not None:
            self.handler.setLevel(self.level)
        root_logger.addHandler(self.handler)
        if self.level is not None:
            self.orig_level = root_logger.level
            root_logger.setLevel(min(self.orig_level, self.level))
        return self.handler

    def __exit__(self, type, value, traceback):
        root_logger = logging.getLogger()
        if self.level is not None:
            root_logger.setLevel(self.orig_level)
        root_logger.removeHandler(self.handler)


class LogCaptureHandler(logging_StreamHandler):
    """A logging handler that stores log records and the log text."""

    def __init__(self) -> None:
        """Create a new log handler."""
        super().__init__(StringIO())
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the log records in a list in addition to the log text."""
        self.records.append(record)
        super().emit(record)

    def reset(self) -> None:
        self.records = []
        self.stream = StringIO()

    def handleError(self, record: logging.LogRecord) -> None:
        if logging.raiseExceptions:
            # Fail the test if the log message is bad (emit failed).
            # The default behavior of logging is to print "Logging error"
            # to stderr with the call stack and some extra details.
            # pytest wants to make such mistakes visible during testing.
            raise


@final
class LogCaptureFixture:
    """Provides access and control of log capturing."""

    def __init__(self, item: nodes.Node, *, _ispytest: bool = False) -> None:
        check_ispytest(_ispytest)
        self._item = item
        self._initial_handler_level: Optional[int] = None
        # Dict of log name -> log level.
        self._initial_logger_levels: Dict[Optional[str], int] = {}

    def _finalize(self) -> None:
        """Finalize the fixture.

        This restores the log levels changed by :meth:`set_level`.
        """
        # Restore log levels.
        if self._initial_handler_level is not None:
            self.handler.setLevel(self._initial_handler_level)
        for logger_name, level in self._initial_logger_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)

    @property
    def handler(self) -> LogCaptureHandler:
        """Get the logging handler used by the fixture.

        :rtype: LogCaptureHandler
        """
        return self._item.stash[caplog_handler_key]

    def get_records(self, when: str) -> List[logging.LogRecord]:
        """Get the logging records for one of the possible test phases.
