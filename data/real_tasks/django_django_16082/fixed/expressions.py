        return hash(self.identity)


# Type inference for CombinedExpression.output_field.
# Missing items will result in FieldError, by design.
#
# The current approach for NULL is based on lowest common denominator behavior
# i.e. if one of the supported databases is raising an error (rather than
# return NULL) for `val <op> NULL`, then Django raises FieldError.
NoneType = type(None)

_connector_combinations = [
    # Numeric operations - operands of same type.
    {
        connector: [
            (fields.IntegerField, fields.IntegerField, fields.IntegerField),
            (fields.FloatField, fields.FloatField, fields.FloatField),
            (fields.DecimalField, fields.DecimalField, fields.DecimalField),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            # Behavior for DIV with integer arguments follows Postgres/SQLite,
            # not MySQL/Oracle.
            Combinable.DIV,
            Combinable.MOD,
            Combinable.POW,
        )
    },
    # Numeric operations - operands of different type.
    {
        connector: [
            (fields.IntegerField, fields.DecimalField, fields.DecimalField),
            (fields.DecimalField, fields.IntegerField, fields.DecimalField),
            (fields.IntegerField, fields.FloatField, fields.FloatField),
            (fields.FloatField, fields.IntegerField, fields.FloatField),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            Combinable.DIV,
            Combinable.MOD,
        )
    },
    # Bitwise operators.
    {
        connector: [
            (fields.IntegerField, fields.IntegerField, fields.IntegerField),
        ]
        for connector in (
            Combinable.BITAND,
            Combinable.BITOR,
            Combinable.BITLEFTSHIFT,
            Combinable.BITRIGHTSHIFT,
            Combinable.BITXOR,
        )
    },
    # Numeric with NULL.
    {
        connector: [
            (field_type, NoneType, field_type),
            (NoneType, field_type, field_type),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            Combinable.DIV,
            Combinable.MOD,
            Combinable.POW,
        )
        for field_type in (fields.IntegerField, fields.DecimalField, fields.FloatField)
    },
    # Date/DateTimeField/DurationField/TimeField.
    {
        Combinable.ADD: [
            # Date/DateTimeField.
            (fields.DateField, fields.DurationField, fields.DateTimeField),
            (fields.DateTimeField, fields.DurationField, fields.DateTimeField),
            (fields.DurationField, fields.DateField, fields.DateTimeField),
            (fields.DurationField, fields.DateTimeField, fields.DateTimeField),
            # DurationField.
            (fields.DurationField, fields.DurationField, fields.DurationField),
            # TimeField.
            (fields.TimeField, fields.DurationField, fields.TimeField),
