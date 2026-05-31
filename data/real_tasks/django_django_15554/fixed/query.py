        self.subq_aliases = self.subq_aliases.union([self.alias_prefix])
        other_query.subq_aliases = other_query.subq_aliases.union(self.subq_aliases)
        if exclude is None:
            exclude = {}
        self.change_aliases(
            {
                alias: "%s%d" % (self.alias_prefix, pos)
                for pos, alias in enumerate(self.alias_map)
                if alias not in exclude
            }
        )

    def get_initial_alias(self):
        """
        Return the first alias for this query, after increasing its reference
        count.
        """
        if self.alias_map:
            alias = self.base_table
            self.ref_alias(alias)
        elif self.model:
            alias = self.join(self.base_table_class(self.get_meta().db_table, None))
        else:
            alias = None
        return alias

    def count_active_tables(self):
        """
        Return the number of tables in this query with a non-zero reference
        count. After execution, the reference counts are zeroed, so tables
        added in compiler will not be seen by this method.
        """
        return len([1 for count in self.alias_refcount.values() if count])

    def join(self, join, reuse=None):
        """
        Return an alias for the 'join', either reusing an existing alias for
        that join or creating a new one. 'join' is either a base_table_class or
        join_class.

        The 'reuse' parameter can be either None which means all joins are
        reusable, or it can be a set containing the aliases that can be reused.

        The 'reuse_with_filtered_relation' parameter is used when computing
        FilteredRelation instances.

        A join is always created as LOUTER if the lhs alias is LOUTER to make
        sure chains like t1 LOUTER t2 INNER t3 aren't generated. All new
        joins are created as LOUTER if the join is nullable.
        """
        if reuse_with_filtered_relation and reuse:
            reuse_aliases = [
                a for a, j in self.alias_map.items() if a in reuse and j.equals(join)
            ]
        else:
            reuse_aliases = [
                a
                for a, j in self.alias_map.items()
                if (reuse is None or a in reuse) and j == join
            ]
        if reuse_aliases:
            if join.table_alias in reuse_aliases:
                reuse_alias = join.table_alias
            else:
                # Reuse the most recent alias of the joined table
                # (a many-to-many relation may be joined multiple times).
                reuse_alias = reuse_aliases[-1]
            self.ref_alias(reuse_alias)
            return reuse_alias

        # No reuse is possible, so we need a new alias.
        alias, _ = self.table_alias(
            join.table_name, create=True, filtered_relation=join.filtered_relation
        )
        if join.join_type:
            if self.alias_map[join.parent_alias].join_type == LOUTER or join.nullable:
                join_type = LOUTER
            else:
                join_type = INNER
            join.join_type = join_type
        join.table_alias = alias
        self.alias_map[alias] = join
        return alias

    def join_parent_model(self, opts, model, alias, seen):
        """
        Make sure the given 'model' is joined in the query. If 'model' isn't
        a parent of 'opts' or if it is None this method is a no-op.

        The 'alias' is the root alias for starting the join, 'seen' is a dict
        of model -> alias of existing joins. It must also contain a mapping
        of None -> some alias. This will be returned in the no-op case.
        """
        if model in seen:
            return seen[model]
        chain = opts.get_base_chain(model)
        if not chain:
            return alias
        curr_opts = opts
        for int_model in chain:
            if int_model in seen:
                curr_opts = int_model._meta
                alias = seen[int_model]
