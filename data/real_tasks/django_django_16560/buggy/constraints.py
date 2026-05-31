        if (
            self.violation_error_message is not None
            and self.violation_error_message != self.default_violation_error_message
        ):
            kwargs["violation_error_message"] = self.violation_error_message
        return (path, (), kwargs)

    def clone(self):
        _, args, kwargs = self.deconstruct()
        return self.__class__(*args, **kwargs)


class CheckConstraint(BaseConstraint):
    def __init__(self, *, check, name, violation_error_message=None):
        self.check = check
        if not getattr(check, "conditional", False):
            raise TypeError(
                "CheckConstraint.check must be a Q instance or boolean expression."
            )
        super().__init__(name=name, violation_error_message=violation_error_message)

    def _get_check_sql(self, model, schema_editor):
        query = Query(model=model, alias_cols=False)
        where = query.build_where(self.check)
        compiler = query.get_compiler(connection=schema_editor.connection)
        sql, params = where.as_sql(compiler, schema_editor.connection)
        return sql % tuple(schema_editor.quote_value(p) for p in params)

    def constraint_sql(self, model, schema_editor):
        check = self._get_check_sql(model, schema_editor)
        return schema_editor._check_sql(self.name, check)

    def create_sql(self, model, schema_editor):
        check = self._get_check_sql(model, schema_editor)
        return schema_editor._create_check_sql(model, self.name, check)

    def remove_sql(self, model, schema_editor):
        return schema_editor._delete_check_sql(model, self.name)

    def validate(self, model, instance, exclude=None, using=DEFAULT_DB_ALIAS):
        against = instance._get_field_value_map(meta=model._meta, exclude=exclude)
        try:
            if not Q(self.check).check(against, using=using):
                raise ValidationError(self.get_violation_error_message())
        except FieldError:
            pass

    def __repr__(self):
        return "<%s: check=%s name=%s%s>" % (
            self.__class__.__qualname__,
            self.check,
            repr(self.name),
            (
                ""
                if self.violation_error_message is None
                or self.violation_error_message == self.default_violation_error_message
                else " violation_error_message=%r" % self.violation_error_message
            ),
        )

    def __eq__(self, other):
        if isinstance(other, CheckConstraint):
            return (
                self.name == other.name
                and self.check == other.check
                and self.violation_error_message == other.violation_error_message
            )
        return super().__eq__(other)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs["check"] = self.check
        return path, args, kwargs


class Deferrable(Enum):
    DEFERRED = "deferred"
    IMMEDIATE = "immediate"

    # A similar format was proposed for Python 3.10.
    def __repr__(self):
        return f"{self.__class__.__qualname__}.{self._name_}"


class UniqueConstraint(BaseConstraint):
    def __init__(
        self,
        *expressions,
        fields=(),
        name=None,
        condition=None,
        deferrable=None,
        include=None,
        opclasses=(),
        violation_error_message=None,
