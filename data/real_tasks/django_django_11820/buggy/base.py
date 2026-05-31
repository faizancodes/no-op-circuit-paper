                    "'ordering' and 'order_with_respect_to' cannot be used together.",
                    obj=cls,
                    id='models.E021',
                ),
            ]

        if cls._meta.order_with_respect_to or not cls._meta.ordering:
            return []

        if not isinstance(cls._meta.ordering, (list, tuple)):
            return [
                checks.Error(
                    "'ordering' must be a tuple or list (even if you want to order by only one field).",
                    obj=cls,
                    id='models.E014',
                )
            ]

        errors = []
        fields = cls._meta.ordering

        # Skip expressions and '?' fields.
        fields = (f for f in fields if isinstance(f, str) and f != '?')

        # Convert "-field" to "field".
        fields = ((f[1:] if f.startswith('-') else f) for f in fields)

        # Separate related fields and non-related fields.
        _fields = []
        related_fields = []
        for f in fields:
            if LOOKUP_SEP in f:
                related_fields.append(f)
            else:
                _fields.append(f)
        fields = _fields

        # Check related fields.
        for field in related_fields:
            _cls = cls
            fld = None
            for part in field.split(LOOKUP_SEP):
                try:
                    fld = _cls._meta.get_field(part)
                    if fld.is_relation:
                        _cls = fld.get_path_info()[-1].to_opts.model
                except (FieldDoesNotExist, AttributeError):
                    if fld is None or fld.get_transform(part) is None:
                        errors.append(
                            checks.Error(
                                "'ordering' refers to the nonexistent field, "
                                "related field, or lookup '%s'." % field,
                                obj=cls,
                                id='models.E015',
                            )
                        )

        # Skip ordering on pk. This is always a valid order_by field
        # but is an alias and therefore won't be found by opts.get_field.
        fields = {f for f in fields if f != 'pk'}

        # Check for invalid or nonexistent fields in ordering.
        invalid_fields = []

        # Any field name that is not present in field_names does not exist.
        # Also, ordering by m2m fields is not allowed.
        opts = cls._meta
        valid_fields = set(chain.from_iterable(
            (f.name, f.attname) if not (f.auto_created and not f.concrete) else (f.field.related_query_name(),)
            for f in chain(opts.fields, opts.related_objects)
        ))

        invalid_fields.extend(fields - valid_fields)

        for invalid_field in invalid_fields:
            errors.append(
                checks.Error(
                    "'ordering' refers to the nonexistent field, related "
                    "field, or lookup '%s'." % invalid_field,
                    obj=cls,
                    id='models.E015',
                )
            )
        return errors

    @classmethod
    def _check_long_column_names(cls):
        """
        Check that any auto-generated column names are shorter than the limits
