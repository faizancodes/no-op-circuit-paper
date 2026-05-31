            return True
        if value in ('f', 'False', '0'):
            return False
        raise exceptions.ValidationError(
            self.error_messages['invalid_nullable' if self.null else 'invalid'],
            code='invalid',
            params={'value': value},
        )

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return None
        return self.to_python(value)

    def formfield(self, **kwargs):
        if self.choices is not None:
            include_blank = not (self.has_default() or 'initial' in kwargs)
            defaults = {'choices': self.get_choices(include_blank=include_blank)}
        else:
            form_class = forms.NullBooleanField if self.null else forms.BooleanField
            # In HTML checkboxes, 'required' means "must be checked" which is
            # different from the choices case ("must select some value").
            # required=False allows unchecked checkboxes.
            defaults = {'form_class': form_class, 'required': False}
        return super().formfield(**{**defaults, **kwargs})

    def select_format(self, compiler, sql, params):
        sql, params = super().select_format(compiler, sql, params)
        # Filters that match everything are handled as empty strings in the
        # WHERE clause, but in SELECT or GROUP BY list they must use a
        # predicate that's always True.
        if sql == '':
            sql = '1'
        return sql, params


class CharField(Field):
    description = _("String (up to %(max_length)s)")

    def __init__(self, *args, db_collation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_collation = db_collation
        if self.max_length is not None:
            self.validators.append(validators.MaxLengthValidator(self.max_length))

    def check(self, **kwargs):
        databases = kwargs.get('databases') or []
        return [
            *super().check(**kwargs),
            *self._check_db_collation(databases),
            *self._check_max_length_attribute(**kwargs),
        ]

    def _check_max_length_attribute(self, **kwargs):
        if self.max_length is None:
            return [
                checks.Error(
                    "CharFields must define a 'max_length' attribute.",
                    obj=self,
                    id='fields.E120',
                )
            ]
        elif (not isinstance(self.max_length, int) or isinstance(self.max_length, bool) or
                self.max_length <= 0):
            return [
                checks.Error(
                    "'max_length' must be a positive integer.",
                    obj=self,
                    id='fields.E121',
                )
            ]
        else:
            return []

    def _check_db_collation(self, databases):
        errors = []
        for db in databases:
            if not router.allow_migrate_model(db, self.model):
                continue
            connection = connections[db]
            if not (
                self.db_collation is None or
                'supports_collation_on_charfield' in self.model._meta.required_db_features or
                connection.features.supports_collation_on_charfield
            ):
                errors.append(
                    checks.Error(
