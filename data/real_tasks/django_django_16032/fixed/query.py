
    # A tuple that is a set of model field names and either True, if these are
    # the fields to defer, or False if these are the only fields to load.
    deferred_loading = (frozenset(), True)

    explain_info = None

    def __init__(self, model, alias_cols=True):
        self.model = model
        self.alias_refcount = {}
        # alias_map is the most important data structure regarding joins.
        # It's used for recording which joins exist in the query and what
        # types they are. The key is the alias of the joined table (possibly
        # the table name) and the value is a Join-like object (see
        # sql.datastructures.Join for more information).
        self.alias_map = {}
        # Whether to provide alias to columns during reference resolving.
        self.alias_cols = alias_cols
        # Sometimes the query contains references to aliases in outer queries (as
        # a result of split_exclude). Correct alias quoting needs to know these
        # aliases too.
        # Map external tables to whether they are aliased.
        self.external_aliases = {}
        self.table_map = {}  # Maps table names to list of aliases.
        self.used_aliases = set()

        self.where = WhereNode()
        # Maps alias -> Annotation Expression.
        self.annotations = {}
        # These are for extensions. The contents are more or less appended
        # verbatim to the appropriate clause.
        self.extra = {}  # Maps col_alias -> (col_sql, params).

        self._filtered_relations = {}

    @property
    def output_field(self):
        if len(self.select) == 1:
            select = self.select[0]
            return getattr(select, "target", None) or select.field
        elif len(self.annotation_select) == 1:
            return next(iter(self.annotation_select.values())).output_field

    @cached_property
    def base_table(self):
        for alias in self.alias_map:
            return alias

    def __str__(self):
        """
        Return the query as a string of SQL with the parameter values
        substituted in (use sql_with_params() to see the unsubstituted string).

        Parameter values won't necessarily be quoted correctly, since that is
        done by the database interface at execution time.
        """
        sql, params = self.sql_with_params()
        return sql % params

    def sql_with_params(self):
        """
        Return the query as an SQL string and the parameters that will be
        substituted into the query.
        """
        return self.get_compiler(DEFAULT_DB_ALIAS).as_sql()

    def __deepcopy__(self, memo):
        """Limit the amount of work when a Query is deepcopied."""
        result = self.clone()
        memo[id(self)] = result
        return result

    def get_compiler(self, using=None, connection=None, elide_empty=True):
        if using is None and connection is None:
            raise ValueError("Need either using or connection")
        if using:
            connection = connections[using]
        return connection.ops.compiler(self.compiler)(
            self, connection, using, elide_empty
        )

    def get_meta(self):
        """
        Return the Options instance (the model._meta) from which to start
        processing. Normally, this is self.model._meta, but it can be changed
        by subclasses.
