            l_method = getattr(
                handler, "leave_%s" % kid, getattr(handler, "leave_default", None)
            )
            self._cache[klass] = (e_method, l_method)
        else:
            e_method, l_method = methods
        return e_method, l_method

    def visit(self, node):
        """walk on the tree from <node>, getting callbacks from handler"""
        method = self.get_callbacks(node)[0]
        if method is not None:
            method(node)

    def leave(self, node):
        """walk on the tree from <node>, getting callbacks from handler"""
        method = self.get_callbacks(node)[1]
        if method is not None:
            method(node)


class LocalsVisitor(ASTWalker):
    """visit a project by traversing the locals dictionary"""

    def __init__(self):
        ASTWalker.__init__(self, self)
        self._visited = set()

    def visit(self, node):
        """launch the visit starting from the given node"""
        if node in self._visited:
            return None

        self._visited.add(node)
        methods = self.get_callbacks(node)
        if methods[0] is not None:
            methods[0](node)
        if hasattr(node, "locals"):  # skip Instance and other proxy
            for local_node in node.values():
                self.visit(local_node)
        if methods[1] is not None:
            return methods[1](node)
        return None


def get_annotation_label(ann: Union[astroid.Name, astroid.Subscript]) -> str:
    label = ""
    if isinstance(ann, astroid.Subscript):
        label = ann.as_string()
    elif isinstance(ann, astroid.Name):
        label = ann.name
    return label


def get_annotation(
    node: Union[astroid.AssignAttr, astroid.AssignName]
) -> Optional[Union[astroid.Name, astroid.Subscript]]:
    """return the annotation for `node`"""
    ann = None
    if isinstance(node.parent, astroid.AnnAssign):
        ann = node.parent.annotation
    elif isinstance(node, astroid.AssignAttr):
        init_method = node.parent.parent
        try:
            annotations = dict(zip(init_method.locals, init_method.args.annotations))
            ann = annotations.get(node.parent.value.name)
        except AttributeError:
            pass
    else:
        return ann

    try:
        default, *_ = node.infer()
    except astroid.InferenceError:
        default = ""

    label = get_annotation_label(ann)
    if ann:
        label = (
            rf"Optional[{label}]"
            if getattr(default, "value", "value") is None
            and not label.startswith("Optional")
            else label
        )
    if label:
        ann.name = label
    return ann


def infer_node(node: Union[astroid.AssignAttr, astroid.AssignName]) -> set:
    """Return a set containing the node annotation if it exists
    otherwise return a set of the inferred types using the NodeNG.infer method"""

    ann = get_annotation(node)
    if ann:
        return {ann}
    try:
        return set(node.infer())
    except astroid.InferenceError:
        return set()
