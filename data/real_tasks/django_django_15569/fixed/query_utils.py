
        found = self._get_lookup(lookup_name)
        if found is None and hasattr(self, "output_field"):
            return self.output_field.get_transform(lookup_name)
        if found is not None and not issubclass(found, Transform):
            return None
        return found

    @staticmethod
    def merge_dicts(dicts):
        """
        Merge dicts in reverse to preference the order of the original list. e.g.,
        merge_dicts([a, b]) will preference the keys in 'a' over those in 'b'.
        """
        merged = {}
        for d in reversed(dicts):
            merged.update(d)
        return merged

    @classmethod
    def _clear_cached_lookups(cls):
        for subclass in subclasses(cls):
            subclass.get_lookups.cache_clear()

    @classmethod
    def register_lookup(cls, lookup, lookup_name=None):
        if lookup_name is None:
            lookup_name = lookup.lookup_name
        if "class_lookups" not in cls.__dict__:
            cls.class_lookups = {}
        cls.class_lookups[lookup_name] = lookup
        cls._clear_cached_lookups()
        return lookup

    @classmethod
    def _unregister_lookup(cls, lookup, lookup_name=None):
        """
        Remove given lookup from cls lookups. For use in tests only as it's
        not thread-safe.
        """
        if lookup_name is None:
            lookup_name = lookup.lookup_name
        del cls.class_lookups[lookup_name]
        cls._clear_cached_lookups()


def select_related_descend(field, restricted, requested, load_fields, reverse=False):
    """
    Return True if this field should be used to descend deeper for
    select_related() purposes. Used by both the query construction code
    (compiler.get_related_selections()) and the model instance creation code
    (compiler.klass_info).

    Arguments:
     * field - the field to be checked
     * restricted - a boolean field, indicating if the field list has been
       manually restricted using a requested clause)
     * requested - The select_related() dictionary.
     * load_fields - the set of fields to be loaded on this model
     * reverse - boolean, True if we are checking a reverse select related
    """
    if not field.remote_field:
        return False
    if field.remote_field.parent_link and not reverse:
        return False
    if restricted:
        if reverse and field.related_query_name() not in requested:
            return False
        if not reverse and field.name not in requested:
            return False
    if not restricted and field.null:
        return False
    if load_fields:
        if field.attname not in load_fields:
            if restricted and field.name in requested:
                msg = (
                    "Field %s.%s cannot be both deferred and traversed using "
                    "select_related at the same time."
                ) % (field.model._meta.object_name, field.name)
                raise FieldError(msg)
    return True


def refs_expression(lookup_parts, annotations):
    """
    Check if the lookup_parts contains references to the given annotations set.
    Because the LOOKUP_SEP is contained in the default annotation names, check
