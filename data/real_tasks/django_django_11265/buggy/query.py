        Trim joins from the start of the join path. The candidates for trim
        are the PathInfos in names_with_path structure that are m2m joins.

        Also set the select column so the start matches the join.

        This method is meant to be used for generating the subquery joins &
        cols in split_exclude().

        Return a lookup usable for doing outerq.filter(lookup=self) and a
        boolean indicating if the joins in the prefix contain a LEFT OUTER join.
        _"""
        all_paths = []
        for _, paths in names_with_path:
            all_paths.extend(paths)
        contains_louter = False
        # Trim and operate only on tables that were generated for
        # the lookup part of the query. That is, avoid trimming
        # joins generated for F() expressions.
        lookup_tables = [
            t for t in self.alias_map
            if t in self._lookup_joins or t == self.base_table
        ]
        for trimmed_paths, path in enumerate(all_paths):
            if path.m2m:
                break
            if self.alias_map[lookup_tables[trimmed_paths + 1]].join_type == LOUTER:
                contains_louter = True
            alias = lookup_tables[trimmed_paths]
            self.unref_alias(alias)
        # The path.join_field is a Rel, lets get the other side's field
        join_field = path.join_field.field
        # Build the filter prefix.
        paths_in_prefix = trimmed_paths
        trimmed_prefix = []
        for name, path in names_with_path:
            if paths_in_prefix - len(path) < 0:
                break
            trimmed_prefix.append(name)
            paths_in_prefix -= len(path)
        trimmed_prefix.append(
            join_field.foreign_related_fields[0].name)
        trimmed_prefix = LOOKUP_SEP.join(trimmed_prefix)
        # Lets still see if we can trim the first join from the inner query
        # (that is, self). We can't do this for LEFT JOINs because we would
        # miss those rows that have nothing on the outer side.
        if self.alias_map[lookup_tables[trimmed_paths + 1]].join_type != LOUTER:
            select_fields = [r[0] for r in join_field.related_fields]
            select_alias = lookup_tables[trimmed_paths + 1]
            self.unref_alias(lookup_tables[trimmed_paths])
            extra_restriction = join_field.get_extra_restriction(
                self.where_class, None, lookup_tables[trimmed_paths + 1])
            if extra_restriction:
                self.where.add(extra_restriction, AND)
        else:
            # TODO: It might be possible to trim more joins from the start of the
            # inner query if it happens to have a longer join chain containing the
            # values in select_fields. Lets punt this one for now.
            select_fields = [r[1] for r in join_field.related_fields]
            select_alias = lookup_tables[trimmed_paths]
        # The found starting point is likely a Join instead of a BaseTable reference.
        # But the first entry in the query's FROM clause must not be a JOIN.
        for table in self.alias_map:
            if self.alias_refcount[table] > 0:
                self.alias_map[table] = BaseTable(self.alias_map[table].table_name, table)
                break
        self.set_select([f.get_col(select_alias) for f in select_fields])
        return trimmed_prefix, contains_louter

    def is_nullable(self, field):
        """
        Check if the given field should be treated as nullable.

        Some backends treat '' as null and Django treats such fields as
        nullable for those backends. In such situations field.null can be
        False even if we should treat the field as nullable.
        """
        # We need to use DEFAULT_DB_ALIAS here, as QuerySet does not have
        # (nor should it have) knowledge of which connection is going to be
        # used. The proper fix would be to defer all decisions where
        # is_nullable() is needed to the compiler stage, but that is not easy
        # to do currently.
        return (
            connections[DEFAULT_DB_ALIAS].features.interprets_empty_strings_as_nulls and
            field.empty_strings_allowed
        ) or field.null


def get_order_dir(field, default='ASC'):
    """
