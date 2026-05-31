        ('diff', 'end-before'),
        ('diff', 'start-at'),
        ('diff', 'end-at'),
    ]

    def __init__(self, filename: str, options: Dict, config: Config) -> None:
        self.filename = filename
        self.options = options
        self.encoding = options.get('encoding', config.source_encoding)
        self.lineno_start = self.options.get('lineno-start', 1)

        self.parse_options()

    def parse_options(self) -> None:
        for option1, option2 in self.INVALID_OPTIONS_PAIR:
            if option1 in self.options and option2 in self.options:
                raise ValueError(__('Cannot use both "%s" and "%s" options') %
                                 (option1, option2))

    def read_file(self, filename: str, location: Tuple[str, int] = None) -> List[str]:
        try:
            with open(filename, encoding=self.encoding, errors='strict') as f:
                text = f.read()
                if 'tab-width' in self.options:
                    text = text.expandtabs(self.options['tab-width'])

                return text.splitlines(True)
        except OSError as exc:
            raise OSError(__('Include file %r not found or reading it failed') %
                          filename) from exc
        except UnicodeError as exc:
            raise UnicodeError(__('Encoding %r used for reading included file %r seems to '
                                  'be wrong, try giving an :encoding: option') %
                               (self.encoding, filename)) from exc

    def read(self, location: Tuple[str, int] = None) -> Tuple[str, int]:
        if 'diff' in self.options:
            lines = self.show_diff()
        else:
            filters = [self.pyobject_filter,
                       self.start_filter,
                       self.end_filter,
                       self.lines_filter,
                       self.prepend_filter,
                       self.append_filter,
                       self.dedent_filter]
            lines = self.read_file(self.filename, location=location)
            for func in filters:
                lines = func(lines, location=location)

        return ''.join(lines), len(lines)

    def show_diff(self, location: Tuple[str, int] = None) -> List[str]:
        new_lines = self.read_file(self.filename)
        old_filename = self.options.get('diff')
        old_lines = self.read_file(old_filename)
        diff = unified_diff(old_lines, new_lines, old_filename, self.filename)
        return list(diff)

    def pyobject_filter(self, lines: List[str], location: Tuple[str, int] = None) -> List[str]:
        pyobject = self.options.get('pyobject')
        if pyobject:
            from sphinx.pycode import ModuleAnalyzer
            analyzer = ModuleAnalyzer.for_file(self.filename, '')
            tags = analyzer.find_tags()
            if pyobject not in tags:
                raise ValueError(__('Object named %r not found in include file %r') %
                                 (pyobject, self.filename))
            else:
                start = tags[pyobject][1]
                end = tags[pyobject][2]
                lines = lines[start - 1:end]
                if 'lineno-match' in self.options:
                    self.lineno_start = start

        return lines

    def lines_filter(self, lines: List[str], location: Tuple[str, int] = None) -> List[str]:
        linespec = self.options.get('lines')
        if linespec:
            linelist = parselinenos(linespec, len(lines))
            if any(i >= len(lines) for i in linelist):
                logger.warning(__('line number spec is out of range(1-%d): %r') %
                               (len(lines), linespec), location=location)

            if 'lineno-match' in self.options:
                # make sure the line list is not "disjoint".
                first = linelist[0]
                if all(first + i == n for i, n in enumerate(linelist)):
