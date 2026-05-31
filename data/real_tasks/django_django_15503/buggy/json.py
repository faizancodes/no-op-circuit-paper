        params = tuple(lhs_params) + tuple(rhs_params)
        return "JSON_CONTAINS(%s, %s)" % (lhs, rhs), params


class ContainedBy(PostgresOperatorLookup):
    lookup_name = "contained_by"
    postgres_operator = "<@"

    def as_sql(self, compiler, connection):
        if not connection.features.supports_json_field_contains:
            raise NotSupportedError(
                "contained_by lookup is not supported on this database backend."
            )
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = tuple(rhs_params) + tuple(lhs_params)
        return "JSON_CONTAINS(%s, %s)" % (rhs, lhs), params


class HasKeyLookup(PostgresOperatorLookup):
    logical_operator = None

    def as_sql(self, compiler, connection, template=None):
        # Process JSON path from the left-hand side.
        if isinstance(self.lhs, KeyTransform):
            lhs, lhs_params, lhs_key_transforms = self.lhs.preprocess_lhs(
                compiler, connection
            )
            lhs_json_path = compile_json_path(lhs_key_transforms)
        else:
            lhs, lhs_params = self.process_lhs(compiler, connection)
            lhs_json_path = "$"
        sql = template % lhs
        # Process JSON path from the right-hand side.
        rhs = self.rhs
        rhs_params = []
        if not isinstance(rhs, (list, tuple)):
            rhs = [rhs]
        for key in rhs:
            if isinstance(key, KeyTransform):
                *_, rhs_key_transforms = key.preprocess_lhs(compiler, connection)
            else:
                rhs_key_transforms = [key]
            rhs_params.append(
                "%s%s"
                % (
                    lhs_json_path,
                    compile_json_path(rhs_key_transforms, include_root=False),
                )
            )
        # Add condition for each key.
        if self.logical_operator:
            sql = "(%s)" % self.logical_operator.join([sql] * len(rhs_params))
        return sql, tuple(lhs_params) + tuple(rhs_params)

    def as_mysql(self, compiler, connection):
        return self.as_sql(
            compiler, connection, template="JSON_CONTAINS_PATH(%s, 'one', %%s)"
        )

    def as_oracle(self, compiler, connection):
        sql, params = self.as_sql(
            compiler, connection, template="JSON_EXISTS(%s, '%%s')"
        )
        # Add paths directly into SQL because path expressions cannot be passed
        # as bind variables on Oracle.
        return sql % tuple(params), []

    def as_postgresql(self, compiler, connection):
        if isinstance(self.rhs, KeyTransform):
            *_, rhs_key_transforms = self.rhs.preprocess_lhs(compiler, connection)
            for key in rhs_key_transforms[:-1]:
                self.lhs = KeyTransform(key, self.lhs)
            self.rhs = rhs_key_transforms[-1]
        return super().as_postgresql(compiler, connection)

    def as_sqlite(self, compiler, connection):
        return self.as_sql(
            compiler, connection, template="JSON_TYPE(%s, %%s) IS NOT NULL"
        )


class HasKey(HasKeyLookup):
    lookup_name = "has_key"
    postgres_operator = "?"
    prepare_rhs = False


class HasKeys(HasKeyLookup):
    lookup_name = "has_keys"
    postgres_operator = "?&"
    logical_operator = " AND "
