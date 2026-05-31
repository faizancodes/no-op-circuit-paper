        if rolename == 'class' and target == 'None':
            # None is not a type, so use obj role instead.
            rolename = 'obj'

        return super().make_xref(rolename, domain, target, innernode, contnode, env)


class PyGroupedField(PyXrefMixin, GroupedField):
    pass


class PyTypedField(PyXrefMixin, TypedField):
    def make_xref(self, rolename: str, domain: str, target: str,
                  innernode: Type[TextlikeNode] = nodes.emphasis,
                  contnode: Node = None, env: BuildEnvironment = None) -> Node:
        if rolename == 'class' and target == 'None':
            # None is not a type, so use obj role instead.
            rolename = 'obj'

        return super().make_xref(rolename, domain, target, innernode, contnode, env)


class PyObject(ObjectDescription):
    """
    Description of a general Python object.

    :cvar allow_nesting: Class is an object that allows for nested namespaces
    :vartype allow_nesting: bool
    """
    option_spec = {
        'noindex': directives.flag,
        'noindexentry': directives.flag,
        'module': directives.unchanged,
        'canonical': directives.unchanged,
        'annotation': directives.unchanged,
    }

    doc_field_types = [
        PyTypedField('parameter', label=_('Parameters'),
                     names=('param', 'parameter', 'arg', 'argument',
                            'keyword', 'kwarg', 'kwparam'),
                     typerolename='class', typenames=('paramtype', 'type'),
                     can_collapse=True),
        PyTypedField('variable', label=_('Variables'), rolename='obj',
                     names=('var', 'ivar', 'cvar'),
                     typerolename='class', typenames=('vartype',),
                     can_collapse=True),
        PyGroupedField('exceptions', label=_('Raises'), rolename='exc',
                       names=('raises', 'raise', 'exception', 'except'),
                       can_collapse=True),
        Field('returnvalue', label=_('Returns'), has_arg=False,
              names=('returns', 'return')),
        PyField('returntype', label=_('Return type'), has_arg=False,
                names=('rtype',), bodyrolename='class'),
    ]

    allow_nesting = False

    def get_signature_prefix(self, sig: str) -> str:
        """May return a prefix to put before the object name in the
        signature.
        """
        return ''

    def needs_arglist(self) -> bool:
        """May return true if an empty argument list is to be generated even if
        the document contains none.
        """
        return False

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        """Transform a Python signature into RST nodes.

        Return (fully qualified name of the thing, classname if any).

        If inside a class, the current class name is handled intelligently:
        * it is stripped from the displayed name if present
        * it is added to the full name (return value) if not present
        """
        m = py_sig_re.match(sig)
        if m is None:
            raise ValueError
        prefix, name, arglist, retann = m.groups()

        # determine module and class name (if applicable), as well as full name
        modname = self.options.get('module', self.env.ref_context.get('py:module'))
        classname = self.env.ref_context.get('py:class')
