    @property
    def external_aliases(self):
        return self.query.external_aliases

    def get_external_cols(self):
        return self.query.get_external_cols()

    def as_sql(self, compiler, connection, template=None, query=None, **extra_context):
        connection.ops.check_expression_support(self)
        template_params = {**self.extra, **extra_context}
        query = query or self.query
        subquery_sql, sql_params = query.as_sql(compiler, connection)
        template_params['subquery'] = subquery_sql[1:-1]

        template = template or template_params.get('template', self.template)
        sql = template % template_params
        return sql, sql_params

    def get_group_by_cols(self, alias=None):
        # If this expression is referenced by an alias for an explicit GROUP BY
        # through values() a reference to this expression and not the
        # underlying .query must be returned to ensure external column
        # references are not grouped against as well.
        if alias:
            return [Ref(alias, self)]
        return self.query.get_group_by_cols()


class Exists(Subquery):
    template = 'EXISTS(%(subquery)s)'
    output_field = fields.BooleanField()

    def __init__(self, queryset, negated=False, **kwargs):
        self.negated = negated
        super().__init__(queryset, **kwargs)

    def __invert__(self):
        clone = self.copy()
        clone.negated = not self.negated
        return clone

    def as_sql(self, compiler, connection, template=None, **extra_context):
        query = self.query.exists(using=connection.alias)
        sql, params = super().as_sql(
            compiler,
            connection,
            template=template,
            query=query,
            **extra_context,
        )
        if self.negated:
            sql = 'NOT {}'.format(sql)
        return sql, params

    def select_format(self, compiler, sql, params):
        # Wrap EXISTS() with a CASE WHEN expression if a database backend
        # (e.g. Oracle) doesn't support boolean expression in SELECT or GROUP
        # BY list.
        if not compiler.connection.features.supports_boolean_expr_in_select_clause:
            sql = 'CASE WHEN {} THEN 1 ELSE 0 END'.format(sql)
        return sql, params


@deconstructible(path='django.db.models.OrderBy')
class OrderBy(Expression):
    template = '%(expression)s %(ordering)s'
    conditional = False

    def __init__(self, expression, descending=False, nulls_first=False, nulls_last=False):
        if nulls_first and nulls_last:
            raise ValueError('nulls_first and nulls_last are mutually exclusive')
        self.nulls_first = nulls_first
        self.nulls_last = nulls_last
        self.descending = descending
        if not hasattr(expression, 'resolve_expression'):
            raise ValueError('expression must be an expression type')
        self.expression = expression

    def __repr__(self):
        return "{}({}, descending={})".format(
            self.__class__.__name__, self.expression, self.descending)

    def set_source_expressions(self, exprs):
        self.expression = exprs[0]

    def get_source_expressions(self):
        return [self.expression]

    def as_sql(self, compiler, connection, template=None, **extra_context):
        template = template or self.template
        if connection.features.supports_order_by_nulls_modifier:
            if self.nulls_last:
                template = '%s NULLS LAST' % template
