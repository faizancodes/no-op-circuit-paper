                # Handle ForeignObjects which can have multiple from_fields/to_fields.
                from_fields = getattr(new_field, 'from_fields', None)
                if from_fields:
                    from_rename_key = (app_label, model_name)
                    new_field.from_fields = tuple([
                        self.renamed_fields.get(from_rename_key + (from_field,), from_field)
                        for from_field in from_fields
                    ])
                    new_field.to_fields = tuple([
                        self.renamed_fields.get(rename_key + (to_field,), to_field)
                        for to_field in new_field.to_fields
                    ])
            if hasattr(new_field, "remote_field") and getattr(new_field.remote_field, "through", None):
                rename_key = (
                    new_field.remote_field.through._meta.app_label,
                    new_field.remote_field.through._meta.model_name,
                )
                if rename_key in self.renamed_models:
                    new_field.remote_field.through = old_field.remote_field.through
            old_field_dec = self.deep_deconstruct(old_field)
            new_field_dec = self.deep_deconstruct(new_field)
            if old_field_dec != new_field_dec:
                both_m2m = old_field.many_to_many and new_field.many_to_many
                neither_m2m = not old_field.many_to_many and not new_field.many_to_many
                if both_m2m or neither_m2m:
                    # Either both fields are m2m or neither is
                    preserve_default = True
                    if (old_field.null and not new_field.null and not new_field.has_default() and
                            not new_field.many_to_many):
                        field = new_field.clone()
                        new_default = self.questioner.ask_not_null_alteration(field_name, model_name)
                        if new_default is not models.NOT_PROVIDED:
                            field.default = new_default
                            preserve_default = False
                    else:
                        field = new_field
                    self.add_operation(
                        app_label,
                        operations.AlterField(
                            model_name=model_name,
                            name=field_name,
                            field=field,
                            preserve_default=preserve_default,
                        )
                    )
                else:
                    # We cannot alter between m2m and concrete fields
                    self._generate_removed_field(app_label, model_name, field_name)
                    self._generate_added_field(app_label, model_name, field_name)

    def create_altered_indexes(self):
        option_name = operations.AddIndex.option_name
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get((app_label, model_name), model_name)
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            old_indexes = old_model_state.options[option_name]
            new_indexes = new_model_state.options[option_name]
            add_idx = [idx for idx in new_indexes if idx not in old_indexes]
            rem_idx = [idx for idx in old_indexes if idx not in new_indexes]

            self.altered_indexes.update({
                (app_label, model_name): {
                    'added_indexes': add_idx, 'removed_indexes': rem_idx,
                }
            })

    def generate_added_indexes(self):
        for (app_label, model_name), alt_indexes in self.altered_indexes.items():
            for index in alt_indexes['added_indexes']:
                self.add_operation(
                    app_label,
                    operations.AddIndex(
                        model_name=model_name,
                        index=index,
                    )
                )

    def generate_removed_indexes(self):
        for (app_label, model_name), alt_indexes in self.altered_indexes.items():
            for index in alt_indexes['removed_indexes']:
                self.add_operation(
                    app_label,
                    operations.RemoveIndex(
                        model_name=model_name,
                        name=index.name,
