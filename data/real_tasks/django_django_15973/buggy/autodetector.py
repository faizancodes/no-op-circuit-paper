
    def generate_removed_constraints(self):
        for (
            app_label,
            model_name,
        ), alt_constraints in self.altered_constraints.items():
            for constraint in alt_constraints["removed_constraints"]:
                self.add_operation(
                    app_label,
                    operations.RemoveConstraint(
                        model_name=model_name,
                        name=constraint.name,
                    ),
                )

    @staticmethod
    def _get_dependencies_for_foreign_key(app_label, model_name, field, project_state):
        remote_field_model = None
        if hasattr(field.remote_field, "model"):
            remote_field_model = field.remote_field.model
        else:
            relations = project_state.relations[app_label, model_name]
            for (remote_app_label, remote_model_name), fields in relations.items():
                if any(
                    field == related_field.remote_field
                    for related_field in fields.values()
                ):
                    remote_field_model = f"{remote_app_label}.{remote_model_name}"
                    break
        # Account for FKs to swappable models
        swappable_setting = getattr(field, "swappable_setting", None)
        if swappable_setting is not None:
            dep_app_label = "__setting__"
            dep_object_name = swappable_setting
        else:
            dep_app_label, dep_object_name = resolve_relation(
                remote_field_model,
                app_label,
                model_name,
            )
        dependencies = [(dep_app_label, dep_object_name, None, True)]
        if getattr(field.remote_field, "through", None):
            through_app_label, through_object_name = resolve_relation(
                remote_field_model,
                app_label,
                model_name,
            )
            dependencies.append((through_app_label, through_object_name, None, True))
        return dependencies

    def _get_altered_foo_together_operations(self, option_name):
        for app_label, model_name in sorted(self.kept_model_keys):
            old_model_name = self.renamed_models.get(
                (app_label, model_name), model_name
            )
            old_model_state = self.from_state.models[app_label, old_model_name]
            new_model_state = self.to_state.models[app_label, model_name]

            # We run the old version through the field renames to account for those
            old_value = old_model_state.options.get(option_name)
            old_value = (
                {
                    tuple(
                        self.renamed_fields.get((app_label, model_name, n), n)
                        for n in unique
                    )
                    for unique in old_value
                }
                if old_value
                else set()
            )

            new_value = new_model_state.options.get(option_name)
            new_value = set(new_value) if new_value else set()

            if old_value != new_value:
                dependencies = []
                for foo_togethers in new_value:
                    for field_name in foo_togethers:
                        field = new_model_state.get_field(field_name)
                        if field.remote_field and field.remote_field.model:
                            dependencies.extend(
                                self._get_dependencies_for_foreign_key(
                                    app_label,
                                    model_name,
                                    field,
                                    self.to_state,
