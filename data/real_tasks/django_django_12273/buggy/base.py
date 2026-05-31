        my_pk = self.pk
        if my_pk is None:
            return self is other
        return my_pk == other.pk

    def __hash__(self):
        if self.pk is None:
            raise TypeError("Model instances without primary key value are unhashable")
        return hash(self.pk)

    def __reduce__(self):
        data = self.__getstate__()
        data[DJANGO_VERSION_PICKLE_KEY] = get_version()
        class_id = self._meta.app_label, self._meta.object_name
        return model_unpickle, (class_id,), data

    def __getstate__(self):
        """Hook to allow choosing the attributes to pickle."""
        return self.__dict__

    def __setstate__(self, state):
        msg = None
        pickled_version = state.get(DJANGO_VERSION_PICKLE_KEY)
        if pickled_version:
            current_version = get_version()
            if current_version != pickled_version:
                msg = (
                    "Pickled model instance's Django version %s does not match "
                    "the current version %s." % (pickled_version, current_version)
                )
        else:
            msg = "Pickled model instance's Django version is not specified."

        if msg:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        self.__dict__.update(state)

    def _get_pk_val(self, meta=None):
        meta = meta or self._meta
        return getattr(self, meta.pk.attname)

    def _set_pk_val(self, value):
        return setattr(self, self._meta.pk.attname, value)

    pk = property(_get_pk_val, _set_pk_val)

    def get_deferred_fields(self):
        """
        Return a set containing names of deferred fields for this instance.
        """
        return {
            f.attname for f in self._meta.concrete_fields
            if f.attname not in self.__dict__
        }

    def refresh_from_db(self, using=None, fields=None):
        """
        Reload field values from the database.

        By default, the reloading happens from the database this instance was
        loaded from, or by the read router if this instance wasn't loaded from
        any database. The using parameter will override the default.

        Fields can be used to specify which fields to reload. The fields
        should be an iterable of field attnames. If fields is None, then
        all non-deferred fields are reloaded.

        When accessing deferred fields of an instance, the deferred loading
        of the field will call this method.
        """
        if fields is None:
            self._prefetched_objects_cache = {}
        else:
            prefetched_objects_cache = getattr(self, '_prefetched_objects_cache', ())
            for field in fields:
                if field in prefetched_objects_cache:
                    del prefetched_objects_cache[field]
                    fields.remove(field)
            if not fields:
                return
            if any(LOOKUP_SEP in f for f in fields):
                raise ValueError(
                    'Found "%s" in fields argument. Relations and transforms '
                    'are not allowed in fields.' % LOOKUP_SEP)
