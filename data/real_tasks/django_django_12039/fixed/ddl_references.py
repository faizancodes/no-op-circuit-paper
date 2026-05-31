
    def __init__(self, table, quote_name):
        self.table = table
        self.quote_name = quote_name

    def references_table(self, table):
        return self.table == table

    def rename_table_references(self, old_table, new_table):
        if self.table == old_table:
            self.table = new_table

    def __str__(self):
        return self.quote_name(self.table)


class TableColumns(Table):
    """Base class for references to multiple columns of a table."""

    def __init__(self, table, columns):
        self.table = table
        self.columns = columns

    def references_column(self, table, column):
        return self.table == table and column in self.columns

    def rename_column_references(self, table, old_column, new_column):
        if self.table == table:
            for index, column in enumerate(self.columns):
                if column == old_column:
                    self.columns[index] = new_column


class Columns(TableColumns):
    """Hold a reference to one or many columns."""

    def __init__(self, table, columns, quote_name, col_suffixes=()):
        self.quote_name = quote_name
        self.col_suffixes = col_suffixes
        super().__init__(table, columns)

    def __str__(self):
        def col_str(column, idx):
            col = self.quote_name(column)
            try:
                suffix = self.col_suffixes[idx]
                if suffix:
                    col = '{} {}'.format(col, suffix)
            except IndexError:
                pass
            return col

        return ', '.join(col_str(column, idx) for idx, column in enumerate(self.columns))


class IndexName(TableColumns):
    """Hold a reference to an index name."""

    def __init__(self, table, columns, suffix, create_index_name):
        self.suffix = suffix
        self.create_index_name = create_index_name
        super().__init__(table, columns)

    def __str__(self):
        return self.create_index_name(self.table, self.columns, self.suffix)


class IndexColumns(Columns):
    def __init__(self, table, columns, quote_name, col_suffixes=(), opclasses=()):
        self.opclasses = opclasses
        super().__init__(table, columns, quote_name, col_suffixes)

    def __str__(self):
        def col_str(column, idx):
            # Index.__init__() guarantees that self.opclasses is the same
            # length as self.columns.
            col = '{} {}'.format(self.quote_name(column), self.opclasses[idx])
            try:
                col = '{} {}'.format(col, self.col_suffixes[idx])
            except IndexError:
                pass
            return col

        return ', '.join(col_str(column, idx) for idx, column in enumerate(self.columns))


class ForeignKeyName(TableColumns):
    """Hold a reference to a foreign key name."""

    def __init__(self, from_table, from_columns, to_table, to_columns, suffix_template, create_fk_name):
        self.to_reference = TableColumns(to_table, to_columns)
        self.suffix_template = suffix_template
        self.create_fk_name = create_fk_name
        super().__init__(from_table, from_columns,)
