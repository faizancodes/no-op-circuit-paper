        )
        if self.returning_fields and self.connection.features.can_return_columns_from_insert:
            if self.connection.features.can_return_rows_from_bulk_insert:
                result.append(self.connection.ops.bulk_insert_sql(fields, placeholder_rows))
                params = param_rows
            else:
                result.append("VALUES (%s)" % ", ".join(placeholder_rows[0]))
                params = [param_rows[0]]
            if ignore_conflicts_suffix_sql:
                result.append(ignore_conflicts_suffix_sql)
            # Skip empty r_sql to allow subclasses to customize behavior for
            # 3rd party backends. Refs #19096.
            r_sql, self.returning_params = self.connection.ops.return_insert_columns(self.returning_fields)
            if r_sql:
                result.append(r_sql)
                params += [self.returning_params]
            return [(" ".join(result), tuple(chain.from_iterable(params)))]

        if can_bulk:
            result.append(self.connection.ops.bulk_insert_sql(fields, placeholder_rows))
            if ignore_conflicts_suffix_sql:
                result.append(ignore_conflicts_suffix_sql)
            return [(" ".join(result), tuple(p for ps in param_rows for p in ps))]
        else:
            if ignore_conflicts_suffix_sql:
                result.append(ignore_conflicts_suffix_sql)
            return [
                (" ".join(result + ["VALUES (%s)" % ", ".join(p)]), vals)
                for p, vals in zip(placeholder_rows, param_rows)
            ]

    def execute_sql(self, returning_fields=None):
        assert not (
            returning_fields and len(self.query.objs) != 1 and
            not self.connection.features.can_return_rows_from_bulk_insert
        )
        self.returning_fields = returning_fields
        with self.connection.cursor() as cursor:
            for sql, params in self.as_sql():
                cursor.execute(sql, params)
            if not self.returning_fields:
                return []
            if self.connection.features.can_return_rows_from_bulk_insert and len(self.query.objs) > 1:
                return self.connection.ops.fetch_returned_insert_rows(cursor)
            if self.connection.features.can_return_columns_from_insert:
                assert len(self.query.objs) == 1
                return [self.connection.ops.fetch_returned_insert_columns(cursor, self.returning_params)]
            return [(self.connection.ops.last_insert_id(
                cursor, self.query.get_meta().db_table, self.query.get_meta().pk.column
            ),)]


class SQLDeleteCompiler(SQLCompiler):
    @cached_property
    def single_alias(self):
        # Ensure base table is in aliases.
        self.query.get_initial_alias()
        return sum(self.query.alias_refcount[t] > 0 for t in self.query.alias_map) == 1

    def _as_sql(self, query):
        result = [
            'DELETE FROM %s' % self.quote_name_unless_alias(query.base_table)
        ]
        where, params = self.compile(query.where)
        if where:
            result.append('WHERE %s' % where)
        return ' '.join(result), tuple(params)

    def as_sql(self):
        """
        Create the SQL for this query. Return the SQL string and list of
        parameters.
        """
        if self.single_alias:
            return self._as_sql(self.query)
        innerq = self.query.clone()
        innerq.__class__ = Query
        innerq.clear_select_clause()
        pk = self.query.model._meta.pk
        innerq.select = [
            pk.get_col(self.query.get_initial_alias())
        ]
        outerq = Query(self.query.model)
        outerq.where = self.query.where_class()
        if not self.connection.features.update_can_self_select:
            # Force the materialization of the inner query to allow reference
            # to the target table on MySQL.
            sql, params = innerq.get_compiler(connection=self.connection).as_sql()
            innerq = RawSQL('SELECT * FROM (%s) subquery' % sql, params)
        outerq.add_q(Q(pk__in=innerq))
        return self._as_sql(outerq)
