                'Transform should be an instance of KeyTransform in order to '
                'use this lookup.'
            )
        key_text_transform = KeyTextTransform(
            key_transform.key_name, *key_transform.source_expressions,
            **key_transform.extra,
        )
        super().__init__(key_text_transform, *args, **kwargs)


class CaseInsensitiveMixin:
    """
    Mixin to allow case-insensitive comparison of JSON values on MySQL.
    MySQL handles strings used in JSON context using the utf8mb4_bin collation.
    Because utf8mb4_bin is a binary collation, comparison of JSON values is
    case-sensitive.
    """
    def process_lhs(self, compiler, connection):
        lhs, lhs_params = super().process_lhs(compiler, connection)
        if connection.vendor == 'mysql':
            return 'LOWER(%s)' % lhs, lhs_params
        return lhs, lhs_params

    def process_rhs(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if connection.vendor == 'mysql':
            return 'LOWER(%s)' % rhs, rhs_params
        return rhs, rhs_params


class KeyTransformIsNull(lookups.IsNull):
    # key__isnull=False is the same as has_key='key'
    def as_oracle(self, compiler, connection):
        if not self.rhs:
            return HasKey(self.lhs.lhs, self.lhs.key_name).as_oracle(compiler, connection)
        return super().as_sql(compiler, connection)

    def as_sqlite(self, compiler, connection):
        if not self.rhs:
            return HasKey(self.lhs.lhs, self.lhs.key_name).as_sqlite(compiler, connection)
        return super().as_sql(compiler, connection)


class KeyTransformExact(JSONExact):
    def process_lhs(self, compiler, connection):
        lhs, lhs_params = super().process_lhs(compiler, connection)
        if connection.vendor == 'sqlite':
            rhs, rhs_params = super().process_rhs(compiler, connection)
            if rhs == '%s' and rhs_params == ['null']:
                lhs, _ = self.lhs.preprocess_lhs(compiler, connection, lhs_only=True)
                lhs = 'JSON_TYPE(%s, %%s)' % lhs
        return lhs, lhs_params

    def process_rhs(self, compiler, connection):
        if isinstance(self.rhs, KeyTransform):
            return super(lookups.Exact, self).process_rhs(compiler, connection)
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if connection.vendor == 'oracle':
            func = []
            for value in rhs_params:
                value = json.loads(value)
                function = 'JSON_QUERY' if isinstance(value, (list, dict)) else 'JSON_VALUE'
                func.append("%s('%s', '$.value')" % (
                    function,
                    json.dumps({'value': value}),
                ))
            rhs = rhs % tuple(func)
            rhs_params = []
        elif connection.vendor == 'sqlite':
            func = ["JSON_EXTRACT(%s, '$')" if value != 'null' else '%s' for value in rhs_params]
            rhs = rhs % tuple(func)
        return rhs, rhs_params

    def as_oracle(self, compiler, connection):
        rhs, rhs_params = super().process_rhs(compiler, connection)
        if rhs_params == ['null']:
            # Field has key and it's NULL.
            has_key_expr = HasKey(self.lhs.lhs, self.lhs.key_name)
            has_key_sql, has_key_params = has_key_expr.as_oracle(compiler, connection)
            is_null_expr = self.lhs.get_lookup('isnull')(self.lhs, True)
            is_null_sql, is_null_params = is_null_expr.as_sql(compiler, connection)
            return (
                '%s AND %s' % (has_key_sql, is_null_sql),
                tuple(has_key_params) + tuple(is_null_params),
            )
        return super().as_sql(compiler, connection)
