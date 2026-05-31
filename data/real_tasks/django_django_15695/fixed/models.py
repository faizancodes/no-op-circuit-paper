                self.model_name_lower,
                models.Index(fields=self.old_fields, name=self.new_name),
            )
            state.remove_model_options(
                app_label,
                self.model_name_lower,
                AlterIndexTogether.option_name,
                self.old_fields,
            )
        else:
            state.rename_index(
                app_label, self.model_name_lower, self.old_name, self.new_name
            )

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if not self.allow_migrate_model(schema_editor.connection.alias, model):
            return

        if self.old_fields:
            from_model = from_state.apps.get_model(app_label, self.model_name)
            columns = [
                from_model._meta.get_field(field).column for field in self.old_fields
            ]
            matching_index_name = schema_editor._constraint_names(
                from_model, column_names=columns, index=True
            )
            if len(matching_index_name) != 1:
                raise ValueError(
                    "Found wrong number (%s) of indexes for %s(%s)."
                    % (
                        len(matching_index_name),
                        from_model._meta.db_table,
                        ", ".join(columns),
                    )
                )
            old_index = models.Index(
                fields=self.old_fields,
                name=matching_index_name[0],
            )
        else:
            from_model_state = from_state.models[app_label, self.model_name_lower]
            old_index = from_model_state.get_index_by_name(self.old_name)
        # Don't alter when the index name is not changed.
        if old_index.name == self.new_name:
            return

        to_model_state = to_state.models[app_label, self.model_name_lower]
        new_index = to_model_state.get_index_by_name(self.new_name)
        schema_editor.rename_index(model, old_index, new_index)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if self.old_fields:
            # Backward operation with unnamed index is a no-op.
            return

        self.new_name_lower, self.old_name_lower = (
            self.old_name_lower,
            self.new_name_lower,
        )
        self.new_name, self.old_name = self.old_name, self.new_name

        self.database_forwards(app_label, schema_editor, from_state, to_state)

        self.new_name_lower, self.old_name_lower = (
            self.old_name_lower,
            self.new_name_lower,
        )
        self.new_name, self.old_name = self.old_name, self.new_name

    def describe(self):
        if self.old_name:
            return (
                f"Rename index {self.old_name} on {self.model_name} to {self.new_name}"
            )
        return (
            f"Rename unnamed index for {self.old_fields} on {self.model_name} to "
            f"{self.new_name}"
        )

    @property
    def migration_name_fragment(self):
        if self.old_name:
            return "rename_%s_%s" % (self.old_name_lower, self.new_name_lower)
        return "rename_%s_%s_%s" % (
            self.model_name_lower,
            "_".join(self.old_fields),
            self.new_name_lower,
        )
