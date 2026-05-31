        if condition or include or opclasses or expressions:
            # Databases support conditional, covering, and functional unique
            # constraints via a unique index.
            sql = self._create_unique_sql(
                model,
                fields,
                name=name,
                condition=condition,
                include=include,
                opclasses=opclasses,
                expressions=expressions,
            )
            if sql:
                self.deferred_sql.append(sql)
            return None
        constraint = self.sql_unique_constraint % {
            'columns': ', '.join(map(self.quote_name, fields)),
            'deferrable': self._deferrable_constraint_sql(deferrable),
        }
        return self.sql_constraint % {
            'name': self.quote_name(name),
            'constraint': constraint,
        }

    def _create_unique_sql(
        self, model, columns, name=None, condition=None, deferrable=None,
        include=None, opclasses=None, expressions=None,
    ):
        if (
            (
                deferrable and
                not self.connection.features.supports_deferrable_unique_constraints
            ) or
            (condition and not self.connection.features.supports_partial_indexes) or
            (include and not self.connection.features.supports_covering_indexes) or
            (expressions and not self.connection.features.supports_expression_indexes)
        ):
            return None

        def create_unique_name(*args, **kwargs):
            return self.quote_name(self._create_index_name(*args, **kwargs))

        compiler = Query(model, alias_cols=False).get_compiler(connection=self.connection)
        table = Table(model._meta.db_table, self.quote_name)
        if name is None:
            name = IndexName(model._meta.db_table, columns, '_uniq', create_unique_name)
        else:
            name = self.quote_name(name)
        if condition or include or opclasses or expressions:
            sql = self.sql_create_unique_index
        else:
            sql = self.sql_create_unique
        if columns:
            columns = self._index_columns(table, columns, col_suffixes=(), opclasses=opclasses)
        else:
            columns = Expressions(model._meta.db_table, expressions, compiler, self.quote_value)
        return Statement(
            sql,
            table=table,
            name=name,
            columns=columns,
            condition=self._index_condition_sql(condition),
            deferrable=self._deferrable_constraint_sql(deferrable),
            include=self._index_include_sql(model, include),
        )

    def _delete_unique_sql(
        self, model, name, condition=None, deferrable=None, include=None,
        opclasses=None, expressions=None,
    ):
        if (
            (
                deferrable and
                not self.connection.features.supports_deferrable_unique_constraints
            ) or
            (condition and not self.connection.features.supports_partial_indexes) or
            (include and not self.connection.features.supports_covering_indexes) or
            (expressions and not self.connection.features.supports_expression_indexes)

        ):
            return None
        if condition or include or opclasses or expressions:
            sql = self.sql_delete_index
        else:
            sql = self.sql_delete_unique
        return self._delete_constraint_sql(sql, model, name)

    def _check_sql(self, name, check):
        return self.sql_constraint % {
