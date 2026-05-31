                # In theory this should only be enforced when dealing with
                # where_parts containing predicates against multi-valued
                # relationships that could affect aggregation results but this
                # is complex to infer properly.
                raise NotImplementedError(
                    "Heterogeneous disjunctive predicates against window functions are "
                    "not implemented when performing conditional aggregation."
                )
        where_node = (
            self.create(where_parts, self.connector, self.negated)
            if where_parts
            else None
        )
        having_node = (
            self.create(having_parts, self.connector, self.negated)
            if having_parts
            else None
        )
        qualify_node = (
            self.create(qualify_parts, self.connector, self.negated)
            if qualify_parts
            else None
        )
        return where_node, having_node, qualify_node

    def as_sql(self, compiler, connection):
        """
        Return the SQL version of the where clause and the value to be
        substituted in. Return '', [] if this node matches everything,
        None, [] if this node is empty, and raise EmptyResultSet if this
        node can't match anything.
        """
        result = []
        result_params = []
        if self.connector == AND:
            full_needed, empty_needed = len(self.children), 1
        else:
            full_needed, empty_needed = 1, len(self.children)

        if self.connector == XOR and not connection.features.supports_logical_xor:
            # Convert if the database doesn't support XOR:
            #   a XOR b XOR c XOR ...
            # to:
            #   (a OR b OR c OR ...) AND MOD(a + b + c + ..., 2) == 1
            # The result of an n-ary XOR is true when an odd number of operands
            # are true.
            lhs = self.__class__(self.children, OR)
            rhs_sum = reduce(
                operator.add,
                (Case(When(c, then=1), default=0) for c in self.children),
            )
            if len(self.children) > 2:
                rhs_sum = Mod(rhs_sum, 2)
            rhs = Exact(1, rhs_sum)
            return self.__class__([lhs, rhs], AND, self.negated).as_sql(
                compiler, connection
            )

        for child in self.children:
            try:
                sql, params = compiler.compile(child)
            except EmptyResultSet:
                empty_needed -= 1
            except FullResultSet:
                full_needed -= 1
            else:
                if sql:
                    result.append(sql)
                    result_params.extend(params)
                else:
                    full_needed -= 1
            # Check if this node matches nothing or everything.
            # First check the amount of full nodes and empty nodes
            # to make this node empty/full.
            # Now, check if this node is full/empty using the
            # counts.
            if empty_needed == 0:
                if self.negated:
                    raise FullResultSet
                else:
                    raise EmptyResultSet
            if full_needed == 0:
                if self.negated:
                    raise EmptyResultSet
                else:
                    raise FullResultSet
        conn = " %s " % self.connector
        sql_string = conn.join(result)
        if not sql_string:
            raise FullResultSet
        if self.negated:
            # Some backends (Oracle at least) need parentheses around the inner
            # SQL in the negated case, even if the inner SQL contains just a
            # single expression.
            sql_string = "NOT (%s)" % sql_string
        elif len(result) > 1 or self.resolved:
