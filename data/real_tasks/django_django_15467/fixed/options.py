                )
            if "choices" not in kwargs:
                kwargs["choices"] = db_field.get_choices(
                    include_blank=db_field.blank, blank_choice=[("", _("None"))]
                )
        return db_field.formfield(**kwargs)

    def get_field_queryset(self, db, db_field, request):
        """
        If the ModelAdmin specifies ordering, the queryset should respect that
        ordering.  Otherwise don't specify the queryset, let the field decide
        (return None in that case).
        """
        related_admin = self.admin_site._registry.get(db_field.remote_field.model)
        if related_admin is not None:
            ordering = related_admin.get_ordering(request)
            if ordering is not None and ordering != ():
                return db_field.remote_field.model._default_manager.using(db).order_by(
                    *ordering
                )
        return None

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        """
        Get a form Field for a ForeignKey.
        """
        db = kwargs.get("using")

        if "widget" not in kwargs:
            if db_field.name in self.get_autocomplete_fields(request):
                kwargs["widget"] = AutocompleteSelect(
                    db_field, self.admin_site, using=db
                )
            elif db_field.name in self.raw_id_fields:
                kwargs["widget"] = widgets.ForeignKeyRawIdWidget(
                    db_field.remote_field, self.admin_site, using=db
                )
            elif db_field.name in self.radio_fields:
                kwargs["widget"] = widgets.AdminRadioSelect(
                    attrs={
                        "class": get_ul_class(self.radio_fields[db_field.name]),
                    }
                )
                kwargs["empty_label"] = (
                    kwargs.get("empty_label", _("None")) if db_field.blank else None
                )

        if "queryset" not in kwargs:
            queryset = self.get_field_queryset(db, db_field, request)
            if queryset is not None:
                kwargs["queryset"] = queryset

        return db_field.formfield(**kwargs)

    def formfield_for_manytomany(self, db_field, request, **kwargs):
        """
        Get a form Field for a ManyToManyField.
        """
        # If it uses an intermediary model that isn't auto created, don't show
        # a field in admin.
        if not db_field.remote_field.through._meta.auto_created:
            return None
        db = kwargs.get("using")

        if "widget" not in kwargs:
            autocomplete_fields = self.get_autocomplete_fields(request)
            if db_field.name in autocomplete_fields:
                kwargs["widget"] = AutocompleteSelectMultiple(
                    db_field,
                    self.admin_site,
                    using=db,
                )
            elif db_field.name in self.raw_id_fields:
                kwargs["widget"] = widgets.ManyToManyRawIdWidget(
                    db_field.remote_field,
                    self.admin_site,
                    using=db,
                )
            elif db_field.name in [*self.filter_vertical, *self.filter_horizontal]:
                kwargs["widget"] = widgets.FilteredSelectMultiple(
                    db_field.verbose_name, db_field.name in self.filter_vertical
                )
        if "queryset" not in kwargs:
            queryset = self.get_field_queryset(db, db_field, request)
            if queryset is not None:
                kwargs["queryset"] = queryset

        form_field = db_field.formfield(**kwargs)
        if isinstance(form_field.widget, SelectMultiple) and not isinstance(
