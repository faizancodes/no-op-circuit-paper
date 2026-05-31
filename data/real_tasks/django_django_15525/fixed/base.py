
    def save_deferred_fields(self, using=None):
        self.m2m_data = {}
        for field, field_value in self.deferred_fields.items():
            opts = self.object._meta
            label = opts.app_label + "." + opts.model_name
            if isinstance(field.remote_field, models.ManyToManyRel):
                try:
                    values = deserialize_m2m_values(
                        field, field_value, using, handle_forward_references=False
                    )
                except M2MDeserializationError as e:
                    raise DeserializationError.WithData(
                        e.original_exc, label, self.object.pk, e.pk
                    )
                self.m2m_data[field.name] = values
            elif isinstance(field.remote_field, models.ManyToOneRel):
                try:
                    value = deserialize_fk_value(
                        field, field_value, using, handle_forward_references=False
                    )
                except Exception as e:
                    raise DeserializationError.WithData(
                        e, label, self.object.pk, field_value
                    )
                setattr(self.object, field.attname, value)
        self.save()


def build_instance(Model, data, db):
    """
    Build a model instance.

    If the model instance doesn't have a primary key and the model supports
    natural keys, try to retrieve it from the database.
    """
    default_manager = Model._meta.default_manager
    pk = data.get(Model._meta.pk.attname)
    if (
        pk is None
        and hasattr(default_manager, "get_by_natural_key")
        and hasattr(Model, "natural_key")
    ):
        obj = Model(**data)
        obj._state.db = db
        natural_key = obj.natural_key()
        try:
            data[Model._meta.pk.attname] = Model._meta.pk.to_python(
                default_manager.db_manager(db).get_by_natural_key(*natural_key).pk
            )
        except Model.DoesNotExist:
            pass
    return Model(**data)


def deserialize_m2m_values(field, field_value, using, handle_forward_references):
    model = field.remote_field.model
    if hasattr(model._default_manager, "get_by_natural_key"):

        def m2m_convert(value):
            if hasattr(value, "__iter__") and not isinstance(value, str):
                return (
                    model._default_manager.db_manager(using)
                    .get_by_natural_key(*value)
                    .pk
                )
            else:
                return model._meta.pk.to_python(value)

    else:

        def m2m_convert(v):
            return model._meta.pk.to_python(v)

    try:
        pks_iter = iter(field_value)
    except TypeError as e:
        raise M2MDeserializationError(e, field_value)
    try:
        values = []
        for pk in pks_iter:
            values.append(m2m_convert(pk))
        return values
    except Exception as e:
        if isinstance(e, ObjectDoesNotExist) and handle_forward_references:
            return DEFER_FIELD
        else:
            raise M2MDeserializationError(e, pk)
