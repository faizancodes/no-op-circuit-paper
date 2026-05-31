                yield Token(TokenType.LPAREN, "(", pos)
                pos += 1
            elif input[pos] == ")":
                yield Token(TokenType.RPAREN, ")", pos)
                pos += 1
            else:
                match = re.match(r"(:?\w|:|\+|-|\.|\[|\])+", input[pos:])
                if match:
                    value = match.group(0)
                    if value == "or":
                        yield Token(TokenType.OR, value, pos)
                    elif value == "and":
                        yield Token(TokenType.AND, value, pos)
                    elif value == "not":
                        yield Token(TokenType.NOT, value, pos)
                    else:
                        yield Token(TokenType.IDENT, value, pos)
                    pos += len(value)
                else:
                    raise ParseError(
                        pos + 1, 'unexpected character "{}"'.format(input[pos]),
                    )
        yield Token(TokenType.EOF, "", pos)

    def accept(self, type: TokenType, *, reject: bool = False) -> Optional[Token]:
        if self.current.type is type:
            token = self.current
            if token.type is not TokenType.EOF:
                self.current = next(self.tokens)
            return token
        if reject:
            self.reject((type,))
        return None

    def reject(self, expected: Sequence[TokenType]) -> "NoReturn":
        raise ParseError(
            self.current.pos + 1,
            "expected {}; got {}".format(
                " OR ".join(type.value for type in expected), self.current.type.value,
            ),
        )


# True, False and None are legal match expression identifiers,
# but illegal as Python identifiers. To fix this, this prefix
# is added to identifiers in the conversion to Python AST.
IDENT_PREFIX = "$"


def expression(s: Scanner) -> ast.Expression:
    if s.accept(TokenType.EOF):
        ret = ast.NameConstant(False)  # type: ast.expr
    else:
        ret = expr(s)
        s.accept(TokenType.EOF, reject=True)
    return ast.fix_missing_locations(ast.Expression(ret))


def expr(s: Scanner) -> ast.expr:
    ret = and_expr(s)
    while s.accept(TokenType.OR):
        rhs = and_expr(s)
        ret = ast.BoolOp(ast.Or(), [ret, rhs])
    return ret


def and_expr(s: Scanner) -> ast.expr:
    ret = not_expr(s)
    while s.accept(TokenType.AND):
        rhs = not_expr(s)
        ret = ast.BoolOp(ast.And(), [ret, rhs])
    return ret


def not_expr(s: Scanner) -> ast.expr:
    if s.accept(TokenType.NOT):
        return ast.UnaryOp(ast.Not(), not_expr(s))
    if s.accept(TokenType.LPAREN):
        ret = expr(s)
        s.accept(TokenType.RPAREN, reject=True)
        return ret
    ident = s.accept(TokenType.IDENT)
    if ident:
        return ast.Name(ident.value, ast.Load())
    s.reject((TokenType.NOT, TokenType.LPAREN, TokenType.IDENT))


class MatcherAdapter(Mapping[str, bool]):
    """Adapts a matcher function to a locals mapping as required by eval()."""

    def __init__(self, matcher: Callable[[str], bool]) -> None:
        self.matcher = matcher
