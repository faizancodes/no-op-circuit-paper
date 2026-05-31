
    @property
    def id_attributes(self):
        return self.config.cpp_id_attributes

    @property
    def paren_attributes(self):
        return self.config.cpp_paren_attributes

    def _parse_string(self) -> str:
        if self.current_char != '"':
            return None
        startPos = self.pos
        self.pos += 1
        escape = False
        while True:
            if self.eof:
                self.fail("Unexpected end during inside string.")
            elif self.current_char == '"' and not escape:
                self.pos += 1
                break
            elif self.current_char == '\\':
                escape = True
            else:
                escape = False
            self.pos += 1
        return self.definition[startPos:self.pos]

    def _parse_literal(self) -> ASTLiteral:
        # -> integer-literal
        #  | character-literal
        #  | floating-literal
        #  | string-literal
        #  | boolean-literal -> "false" | "true"
        #  | pointer-literal -> "nullptr"
        #  | user-defined-literal
        self.skip_ws()
        if self.skip_word('nullptr'):
            return ASTPointerLiteral()
        if self.skip_word('true'):
            return ASTBooleanLiteral(True)
        if self.skip_word('false'):
            return ASTBooleanLiteral(False)
        pos = self.pos
        if self.match(float_literal_re):
            hasSuffix = self.match(float_literal_suffix_re)
            floatLit = ASTNumberLiteral(self.definition[pos:self.pos])
            if hasSuffix:
                return floatLit
            else:
                return _udl(floatLit)
        for regex in [binary_literal_re, hex_literal_re,
                      integer_literal_re, octal_literal_re]:
            if self.match(regex):
                hasSuffix = self.match(integers_literal_suffix_re)
                intLit = ASTNumberLiteral(self.definition[pos:self.pos])
                if hasSuffix:
                    return intLit
                else:
                    return _udl(intLit)

        string = self._parse_string()
        if string is not None:
            return _udl(ASTStringLiteral(string))

        # character-literal
        if self.match(char_literal_re):
            prefix = self.last_match.group(1)  # may be None when no prefix
            data = self.last_match.group(2)
            try:
                charLit = ASTCharLiteral(prefix, data)
            except UnicodeDecodeError as e:
                self.fail("Can not handle character literal. Internal error was: %s" % e)
            except UnsupportedMultiCharacterCharLiteral:
                self.fail("Can not handle character literal"
                          " resulting in multiple decoded characters.")
            return _udl(charLit)
        return None

    def _parse_fold_or_paren_expression(self) -> ASTExpression:
        # "(" expression ")"
        # fold-expression
        # -> ( cast-expression fold-operator ... )
        #  | ( ... fold-operator cast-expression )
        #  | ( cast-expression fold-operator ... fold-operator cast-expression
        if self.current_char != '(':
            return None
        self.pos += 1
        self.skip_ws()
        if self.skip_string_and_ws("..."):
            # ( ... fold-operator cast-expression )
            if not self.match(_fold_operator_re):
                self.fail("Expected fold operator after '...' in fold expression.")
            op = self.matched_text
            rightExpr = self._parse_cast_expression()
            if not self.skip_string(')'):
                self.fail("Expected ')' in end of fold expression.")
            return ASTFoldExpr(None, op, rightExpr)
        # try first parsing a unary right fold, or a binary fold
        pos = self.pos
        try:
            self.skip_ws()
            leftExpr = self._parse_cast_expression()
            self.skip_ws()
            if not self.match(_fold_operator_re):
                self.fail("Expected fold operator after left expression in fold expression.")
            op = self.matched_text
            self.skip_ws()
            if not self.skip_string_and_ws('...'):
                self.fail("Expected '...' after fold operator in fold expression.")
        except DefinitionError as eFold:
            self.pos = pos
            # fall back to a paren expression
            try:
                res = self._parse_expression()
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail("Expected ')' in end of parenthesized expression.")
            except DefinitionError as eExpr:
                raise self._make_multi_error([
