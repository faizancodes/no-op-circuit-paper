        clone.bump_prefix(query)
        clone.subquery = True
        # It's safe to drop ordering if the queryset isn't using slicing,
        # distinct(*fields) or select_for_update().
        if (self.low_mark == 0 and self.high_mark is None and
                not self.distinct_fields and
                not self.select_for_update):
            clone.clear_ordering(True)
        clone.where.resolve_expression(query, *args, **kwargs)
        for key, value in clone.annotations.items():
            resolved = value.resolve_expression(query, *args, **kwargs)
            if hasattr(resolved, 'external_aliases'):
                resolved.external_aliases.update(clone.external_aliases)
            clone.annotations[key] = resolved
        # Outer query's aliases are considered external.
        for alias, table in query.alias_map.items():
            clone.external_aliases[alias] = (
                (isinstance(table, Join) and table.join_field.related_model._meta.db_table != alias) or
                (isinstance(table, BaseTable) and table.table_name != table.table_alias)
            )
        return clone

    def get_external_cols(self):
        exprs = chain(self.annotations.values(), self.where.children)
        return [
            col for col in self._gen_cols(exprs)
            if col.alias in self.external_aliases
        ]

    def as_sql(self, compiler, connection):
        sql, params = self.get_compiler(connection=connection).as_sql()
        if self.subquery:
            sql = '(%s)' % sql
        return sql, params

    def resolve_lookup_value(self, value, can_reuse, allow_joins):
        if hasattr(value, 'resolve_expression'):
            value = value.resolve_expression(
                self, reuse=can_reuse, allow_joins=allow_joins,
            )
        elif isinstance(value, (list, tuple)):
            # The items of the iterable may be expressions and therefore need
            # to be resolved independently.
            return type(value)(
                self.resolve_lookup_value(sub_value, can_reuse, allow_joins)
                for sub_value in value
            )
        return value

    def solve_lookup_type(self, lookup):
        """
        Solve the lookup type from the lookup (e.g.: 'foobar__id__icontains').
        """
        lookup_splitted = lookup.split(LOOKUP_SEP)
        if self.annotations:
            expression, expression_lookups = refs_expression(lookup_splitted, self.annotations)
            if expression:
                return expression_lookups, (), expression
        _, field, _, lookup_parts = self.names_to_path(lookup_splitted, self.get_meta())
        field_parts = lookup_splitted[0:len(lookup_splitted) - len(lookup_parts)]
        if len(lookup_parts) > 1 and not field_parts:
            raise FieldError(
                'Invalid lookup "%s" for model %s".' %
                (lookup, self.get_meta().model.__name__)
            )
        return lookup_parts, field_parts, False

    def check_query_object_type(self, value, opts, field):
        """
        Check whether the object passed while querying is of the correct type.
        If not, raise a ValueError specifying the wrong object.
        """
        if hasattr(value, '_meta'):
            if not check_rel_lookup_compatibility(value._meta.model, opts, field):
                raise ValueError(
                    'Cannot query "%s": Must be "%s" instance.' %
                    (value, opts.object_name))

    def check_related_objects(self, field, value, opts):
        """Check the type of object passed to query relations."""
        if field.is_relation:
            # Check that the field and the queryset use the same model in a
            # query like .filter(author=Author.objects.all()). For example, the
            # opts would be Author's (from the author field) and value.model
            # would be Author.objects.all() queryset's .model (Author also).
            # The field is the related field on the lhs side.
            if (isinstance(value, Query) and not value.has_select_fields and
                    not check_rel_lookup_compatibility(value.model, opts, field)):
                raise ValueError(
                    'Cannot use QuerySet for "%s": Use a QuerySet for "%s".' %
