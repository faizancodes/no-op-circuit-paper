
    def remove_index(self, model, index):
        """Remove an index from a model."""
        self.execute(index.remove_sql(model, self))

    def add_constraint(self, model, constraint):
        """Add a constraint to a model."""
        sql = constraint.create_sql(model, self)
        if sql:
            self.execute(sql)

    def remove_constraint(self, model, constraint):
        """Remove a constraint from a model."""
        sql = constraint.remove_sql(model, self)
        if sql:
            self.execute(sql)

    def alter_unique_together(self, model, old_unique_together, new_unique_together):
        """
        Deal with a model changing its unique_together. The input
        unique_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_unique_together}
        news = {tuple(fields) for fields in new_unique_together}
        # Deleted uniques
        for fields in olds.difference(news):
            self._delete_composed_index(model, fields, {'unique': True}, self.sql_delete_unique)
        # Created uniques
        for fields in news.difference(olds):
            columns = [model._meta.get_field(field).column for field in fields]
            self.execute(self._create_unique_sql(model, columns))

    def alter_index_together(self, model, old_index_together, new_index_together):
        """
        Deal with a model changing its index_together. The input
        index_togethers must be doubly-nested, not the single-nested
        ["foo", "bar"] format.
        """
        olds = {tuple(fields) for fields in old_index_together}
        news = {tuple(fields) for fields in new_index_together}
        # Deleted indexes
        for fields in olds.difference(news):
            self._delete_composed_index(
                model,
                fields,
                {'index': True, 'unique': False},
                self.sql_delete_index,
            )
        # Created indexes
        for field_names in news.difference(olds):
            fields = [model._meta.get_field(field) for field in field_names]
            self.execute(self._create_index_sql(model, fields, suffix="_idx"))

    def _delete_composed_index(self, model, fields, constraint_kwargs, sql):
        meta_constraint_names = {constraint.name for constraint in model._meta.constraints}
        meta_index_names = {constraint.name for constraint in model._meta.indexes}
        columns = [model._meta.get_field(field).column for field in fields]
        constraint_names = self._constraint_names(
            model, columns, exclude=meta_constraint_names | meta_index_names,
            **constraint_kwargs
        )
        if len(constraint_names) != 1:
            raise ValueError("Found wrong number (%s) of constraints for %s(%s)" % (
                len(constraint_names),
                model._meta.db_table,
                ", ".join(columns),
            ))
        self.execute(self._delete_constraint_sql(sql, model, constraint_names[0]))

    def alter_db_table(self, model, old_db_table, new_db_table):
        """Rename the table a model points to."""
        if (old_db_table == new_db_table or
            (self.connection.features.ignores_table_name_case and
                old_db_table.lower() == new_db_table.lower())):
            return
        self.execute(self.sql_rename_table % {
            "old_table": self.quote_name(old_db_table),
            "new_table": self.quote_name(new_db_table),
        })
        # Rename all references to the old table name.
        for sql in self.deferred_sql:
            if isinstance(sql, Statement):
                sql.rename_table_references(old_db_table, new_db_table)

    def alter_db_tablespace(self, model, old_db_tablespace, new_db_tablespace):
        """Move a model's table between tablespaces."""
        self.execute(self.sql_retablespace_table % {
            "table": self.quote_name(model._meta.db_table),
            "old_tablespace": self.quote_name(old_db_tablespace),
            "new_tablespace": self.quote_name(new_db_tablespace),
        })
