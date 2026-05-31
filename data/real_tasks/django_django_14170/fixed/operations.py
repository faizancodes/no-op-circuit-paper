    def adapt_datefield_value(self, value):
        """
        Transform a date value to an object compatible with what is expected
        by the backend driver for date columns.
        """
        if value is None:
            return None
        return str(value)

    def adapt_datetimefield_value(self, value):
        """
        Transform a datetime value to an object compatible with what is expected
        by the backend driver for datetime columns.
        """
        if value is None:
            return None
        return str(value)

    def adapt_timefield_value(self, value):
        """
        Transform a time value to an object compatible with what is expected
        by the backend driver for time columns.
        """
        if value is None:
            return None
        if timezone.is_aware(value):
            raise ValueError("Django does not support timezone-aware times.")
        return str(value)

    def adapt_decimalfield_value(self, value, max_digits=None, decimal_places=None):
        """
        Transform a decimal.Decimal value to an object compatible with what is
        expected by the backend driver for decimal (numeric) columns.
        """
        return utils.format_number(value, max_digits, decimal_places)

    def adapt_ipaddressfield_value(self, value):
        """
        Transform a string representation of an IP address into the expected
        type for the backend driver.
        """
        return value or None

    def year_lookup_bounds_for_date_field(self, value, iso_year=False):
        """
        Return a two-elements list with the lower and upper bound to be used
        with a BETWEEN operator to query a DateField value using a year
        lookup.

        `value` is an int, containing the looked-up year.
        If `iso_year` is True, return bounds for ISO-8601 week-numbering years.
        """
        if iso_year:
            first = datetime.date.fromisocalendar(value, 1, 1)
            second = (
                datetime.date.fromisocalendar(value + 1, 1, 1) -
                datetime.timedelta(days=1)
            )
        else:
            first = datetime.date(value, 1, 1)
            second = datetime.date(value, 12, 31)
        first = self.adapt_datefield_value(first)
        second = self.adapt_datefield_value(second)
        return [first, second]

    def year_lookup_bounds_for_datetime_field(self, value, iso_year=False):
        """
        Return a two-elements list with the lower and upper bound to be used
        with a BETWEEN operator to query a DateTimeField value using a year
        lookup.

        `value` is an int, containing the looked-up year.
        If `iso_year` is True, return bounds for ISO-8601 week-numbering years.
        """
        if iso_year:
            first = datetime.datetime.fromisocalendar(value, 1, 1)
            second = (
                datetime.datetime.fromisocalendar(value + 1, 1, 1) -
                datetime.timedelta(microseconds=1)
            )
        else:
            first = datetime.datetime(value, 1, 1)
            second = datetime.datetime(value, 12, 31, 23, 59, 59, 999999)
        if settings.USE_TZ:
            tz = timezone.get_current_timezone()
            first = timezone.make_aware(first, tz)
            second = timezone.make_aware(second, tz)
        first = self.adapt_datetimefield_value(first)
        second = self.adapt_datetimefield_value(second)
        return [first, second]

    def get_db_converters(self, expression):
        """
        Return a list of functions needed to convert field data.

        Some field types on some backends do not provide data in the correct
        format, this is the hook for converter functions.
        """
        return []

    def convert_durationfield_value(self, value, expression, connection):
        if value is not None:
            return datetime.timedelta(0, 0, value)

    def check_expression_support(self, expression):
        """
        Check that the backend supports the provided expression.

        This is used on specific backends to rule out known expressions
        that have problematic or nonexistent implementations. If the
        expression has a known problem, the backend should raise
        NotSupportedError.
        """
        pass

    def conditional_expression_supported_in_where_clause(self, expression):
        """
        Return True, if the conditional expression is supported in the WHERE
        clause.
        """
        return True

    def combine_expression(self, connector, sub_expressions):
        """
        Combine a list of subexpressions into a single expression, using
        the provided connecting operator. This is required because operators
