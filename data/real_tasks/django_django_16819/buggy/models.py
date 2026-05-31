        self.index = index

    def state_forwards(self, app_label, state):
        state.add_index(app_label, self.model_name_lower, self.index)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.add_index(model, self.index)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            schema_editor.remove_index(model, self.index)

    def deconstruct(self):
        kwargs = {
            "model_name": self.model_name,
            "index": self.index,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs,
        )

    def describe(self):
        if self.index.expressions:
            return "Create index %s on %s on model %s" % (
                self.index.name,
                ", ".join([str(expression) for expression in self.index.expressions]),
                self.model_name,
            )
        return "Create index %s on field(s) %s of model %s" % (
            self.index.name,
            ", ".join(self.index.fields),
            self.model_name,
        )

    @property
    def migration_name_fragment(self):
        return "%s_%s" % (self.model_name_lower, self.index.name.lower())


class RemoveIndex(IndexOperation):
    """Remove an index from a model."""

    def __init__(self, model_name, name):
        self.model_name = model_name
        self.name = name

    def state_forwards(self, app_label, state):
        state.remove_index(app_label, self.model_name_lower, self.name)

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            from_model_state = from_state.models[app_label, self.model_name_lower]
            index = from_model_state.get_index_by_name(self.name)
            schema_editor.remove_index(model, index)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.model_name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            to_model_state = to_state.models[app_label, self.model_name_lower]
            index = to_model_state.get_index_by_name(self.name)
            schema_editor.add_index(model, index)

    def deconstruct(self):
        kwargs = {
            "model_name": self.model_name,
            "name": self.name,
        }
        return (
            self.__class__.__qualname__,
            [],
            kwargs,
        )

    def describe(self):
        return "Remove index %s from %s" % (self.name, self.model_name)

    @property
    def migration_name_fragment(self):
        return "remove_%s_%s" % (self.model_name_lower, self.name.lower())
