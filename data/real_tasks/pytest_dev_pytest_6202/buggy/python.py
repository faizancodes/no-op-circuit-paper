    instance = pyobj_property("Instance")


class PyobjMixin(PyobjContext):
    _ALLOW_MARKERS = True

    @property
    def obj(self):
        """Underlying Python object."""
        obj = getattr(self, "_obj", None)
        if obj is None:
            self._obj = obj = self._getobj()
            # XXX evil hack
            # used to avoid Instance collector marker duplication
            if self._ALLOW_MARKERS:
                self.own_markers.extend(get_unpacked_marks(self.obj))
        return obj

    @obj.setter
    def obj(self, value):
        self._obj = value

    def _getobj(self):
        """Gets the underlying Python object. May be overwritten by subclasses."""
        return getattr(self.parent.obj, self.name)

    def getmodpath(self, stopatmodule=True, includemodule=False):
        """ return python path relative to the containing module. """
        chain = self.listchain()
        chain.reverse()
        parts = []
        for node in chain:
            if isinstance(node, Instance):
                continue
            name = node.name
            if isinstance(node, Module):
                name = os.path.splitext(name)[0]
                if stopatmodule:
                    if includemodule:
                        parts.append(name)
                    break
            parts.append(name)
        parts.reverse()
        s = ".".join(parts)
        return s.replace(".[", "[")

    def reportinfo(self):
        # XXX caching?
        obj = self.obj
        compat_co_firstlineno = getattr(obj, "compat_co_firstlineno", None)
        if isinstance(compat_co_firstlineno, int):
            # nose compatibility
            fspath = sys.modules[obj.__module__].__file__
            if fspath.endswith(".pyc"):
                fspath = fspath[:-1]
            lineno = compat_co_firstlineno
        else:
            fspath, lineno = getfslineno(obj)
        modpath = self.getmodpath()
        assert isinstance(lineno, int)
        return fspath, lineno, modpath


class PyCollector(PyobjMixin, nodes.Collector):
    def funcnamefilter(self, name):
        return self._matches_prefix_or_glob_option("python_functions", name)

    def isnosetest(self, obj):
        """ Look for the __test__ attribute, which is applied by the
        @nose.tools.istest decorator
        """
        # We explicitly check for "is True" here to not mistakenly treat
        # classes with a custom __getattr__ returning something truthy (like a
        # function) as test classes.
        return safe_getattr(obj, "__test__", False) is True

    def classnamefilter(self, name):
        return self._matches_prefix_or_glob_option("python_classes", name)

    def istestfunction(self, obj, name):
        if self.funcnamefilter(name) or self.isnosetest(obj):
            if isinstance(obj, staticmethod):
                # static methods need to be unwrapped
                obj = safe_getattr(obj, "__func__", False)
            return (
                safe_getattr(obj, "__call__", False)
                and fixtures.getfixturemarker(obj) is None
            )
