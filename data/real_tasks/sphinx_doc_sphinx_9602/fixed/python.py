
def _parse_annotation(annotation: str, env: BuildEnvironment = None) -> List[Node]:
    """Parse type annotation."""
    def unparse(node: ast.AST) -> List[Node]:
        if isinstance(node, ast.Attribute):
            return [nodes.Text("%s.%s" % (unparse(node.value)[0], node.attr))]
        elif isinstance(node, ast.BinOp):
            result: List[Node] = unparse(node.left)
            result.extend(unparse(node.op))
            result.extend(unparse(node.right))
            return result
        elif isinstance(node, ast.BitOr):
            return [nodes.Text(' '), addnodes.desc_sig_punctuation('', '|'), nodes.Text(' ')]
        elif isinstance(node, ast.Constant):  # type: ignore
            if node.value is Ellipsis:
                return [addnodes.desc_sig_punctuation('', "...")]
            else:
                return [nodes.Text(node.value)]
        elif isinstance(node, ast.Expr):
            return unparse(node.value)
        elif isinstance(node, ast.Index):
            return unparse(node.value)
        elif isinstance(node, ast.List):
            result = [addnodes.desc_sig_punctuation('', '[')]
            if node.elts:
                # check if there are elements in node.elts to only pop the
                # last element of result if the for-loop was run at least
                # once
                for elem in node.elts:
                    result.extend(unparse(elem))
                    result.append(addnodes.desc_sig_punctuation('', ', '))
                result.pop()
            result.append(addnodes.desc_sig_punctuation('', ']'))
            return result
        elif isinstance(node, ast.Module):
            return sum((unparse(e) for e in node.body), [])
        elif isinstance(node, ast.Name):
            return [nodes.Text(node.id)]
        elif isinstance(node, ast.Subscript):
            result = unparse(node.value)
            result.append(addnodes.desc_sig_punctuation('', '['))
            result.extend(unparse(node.slice))
            result.append(addnodes.desc_sig_punctuation('', ']'))

            # Wrap the Text nodes inside brackets by literal node if the subscript is a Literal
            if result[0] in ('Literal', 'typing.Literal'):
                for i, subnode in enumerate(result[1:], start=1):
                    if isinstance(subnode, nodes.Text):
                        result[i] = nodes.literal('', '', subnode)
            return result
        elif isinstance(node, ast.Tuple):
            if node.elts:
                result = []
                for elem in node.elts:
                    result.extend(unparse(elem))
                    result.append(addnodes.desc_sig_punctuation('', ', '))
                result.pop()
            else:
                result = [addnodes.desc_sig_punctuation('', '('),
                          addnodes.desc_sig_punctuation('', ')')]

            return result
        else:
            if sys.version_info < (3, 8):
                if isinstance(node, ast.Ellipsis):
                    return [addnodes.desc_sig_punctuation('', "...")]
                elif isinstance(node, ast.NameConstant):
                    return [nodes.Text(node.value)]

            raise SyntaxError  # unsupported syntax

    if env is None:
        warnings.warn("The env parameter for _parse_annotation becomes required now.",
                      RemovedInSphinx50Warning, stacklevel=2)

    try:
        tree = ast_parse(annotation)
        result = unparse(tree)
        for i, node in enumerate(result):
            if isinstance(node, nodes.Text) and node.strip():
                result[i] = type_to_xref(str(node), env)
        return result
    except SyntaxError:
        return [type_to_xref(annotation, env)]


def _parse_arglist(arglist: str, env: BuildEnvironment = None) -> addnodes.desc_parameterlist:
    """Parse a list of arguments using AST parser"""
    params = addnodes.desc_parameterlist(arglist)
    sig = signature_from_str('(%s)' % arglist)
    last_kind = None
    for param in sig.parameters.values():
