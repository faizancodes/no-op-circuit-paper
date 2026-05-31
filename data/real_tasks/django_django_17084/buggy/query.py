            obj.__class__ = klass
        if not obj.filter_is_sticky:
            obj.used_aliases = set()
        obj.filter_is_sticky = False
        if hasattr(obj, "_setup_query"):
            obj._setup_query()
        return obj

    def relabeled_clone(self, change_map):
        clone = self.clone()
        clone.change_aliases(change_map)
        return clone

    def _get_col(self, target, field, alias):
        if not self.alias_cols:
            alias = None
        return target.get_col(alias, field)

    def get_aggregation(self, using, aggregate_exprs):
        """
        Return the dictionary with the values of the existing aggregations.
        """
        if not aggregate_exprs:
            return {}
        # Store annotation mask prior to temporarily adding aggregations for
        # resolving purpose to facilitate their subsequent removal.
        refs_subquery = False
        replacements = {}
        annotation_select_mask = self.annotation_select_mask
        for alias, aggregate_expr in aggregate_exprs.items():
            self.check_alias(alias)
            aggregate = aggregate_expr.resolve_expression(
                self, allow_joins=True, reuse=None, summarize=True
            )
            if not aggregate.contains_aggregate:
                raise TypeError("%s is not an aggregate expression" % alias)
            # Temporarily add aggregate to annotations to allow remaining
            # members of `aggregates` to resolve against each others.
            self.append_annotation_mask([alias])
            refs_subquery |= any(
                getattr(self.annotations[ref], "subquery", False)
                for ref in aggregate.get_refs()
            )
            aggregate = aggregate.replace_expressions(replacements)
            self.annotations[alias] = aggregate
            replacements[Ref(alias, aggregate)] = aggregate
        # Stash resolved aggregates now that they have been allowed to resolve
        # against each other.
        aggregates = {alias: self.annotations.pop(alias) for alias in aggregate_exprs}
        self.set_annotation_mask(annotation_select_mask)
        # Existing usage of aggregation can be determined by the presence of
        # selected aggregates but also by filters against aliased aggregates.
        _, having, qualify = self.where.split_having_qualify()
        has_existing_aggregation = (
            any(
                getattr(annotation, "contains_aggregate", True)
                for annotation in self.annotations.values()
            )
            or having
        )
        # Decide if we need to use a subquery.
        #
        # Existing aggregations would cause incorrect results as
        # get_aggregation() must produce just one result and thus must not use
        # GROUP BY.
        #
        # If the query has limit or distinct, or uses set operations, then
        # those operations must be done in a subquery so that the query
        # aggregates on the limit and/or distinct results instead of applying
        # the distinct and limit after the aggregation.
        if (
            isinstance(self.group_by, tuple)
            or self.is_sliced
            or has_existing_aggregation
            or refs_subquery
            or qualify
            or self.distinct
            or self.combinator
        ):
            from django.db.models.sql.subqueries import AggregateQuery

            inner_query = self.clone()
            inner_query.subquery = True
            outer_query = AggregateQuery(self.model, inner_query)
            inner_query.select_for_update = False
            inner_query.select_related = False
