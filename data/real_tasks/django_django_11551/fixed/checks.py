                )
            ]

        from django.contrib.admin.options import InlineModelAdmin

        if not _issubclass(inline, InlineModelAdmin):
            return [
                checks.Error(
                    "'%s' must inherit from 'InlineModelAdmin'." % inline_label,
                    obj=obj.__class__,
                    id='admin.E104',
                )
            ]
        elif not inline.model:
            return [
                checks.Error(
                    "'%s' must have a 'model' attribute." % inline_label,
                    obj=obj.__class__,
                    id='admin.E105',
                )
            ]
        elif not _issubclass(inline.model, models.Model):
            return must_be('a Model', option='%s.model' % inline_label, obj=obj, id='admin.E106')
        else:
            return inline(obj.model, obj.admin_site).check()

    def _check_list_display(self, obj):
        """ Check that list_display only contains fields or usable attributes.
        """

        if not isinstance(obj.list_display, (list, tuple)):
            return must_be('a list or tuple', option='list_display', obj=obj, id='admin.E107')
        else:
            return list(chain.from_iterable(
                self._check_list_display_item(obj, item, "list_display[%d]" % index)
                for index, item in enumerate(obj.list_display)
            ))

    def _check_list_display_item(self, obj, item, label):
        if callable(item):
            return []
        elif hasattr(obj, item):
            return []
        try:
            field = obj.model._meta.get_field(item)
        except FieldDoesNotExist:
            try:
                field = getattr(obj.model, item)
            except AttributeError:
                return [
                    checks.Error(
                        "The value of '%s' refers to '%s', which is not a "
                        "callable, an attribute of '%s', or an attribute or "
                        "method on '%s.%s'." % (
                            label, item, obj.__class__.__name__,
                            obj.model._meta.app_label, obj.model._meta.object_name,
                        ),
                        obj=obj.__class__,
                        id='admin.E108',
                    )
                ]
        if isinstance(field, models.ManyToManyField):
            return [
                checks.Error(
                    "The value of '%s' must not be a ManyToManyField." % label,
                    obj=obj.__class__,
                    id='admin.E109',
                )
            ]
        return []

    def _check_list_display_links(self, obj):
        """ Check that list_display_links is a unique subset of list_display.
        """
        from django.contrib.admin.options import ModelAdmin

        if obj.list_display_links is None:
            return []
        elif not isinstance(obj.list_display_links, (list, tuple)):
            return must_be('a list, a tuple, or None', option='list_display_links', obj=obj, id='admin.E110')
        # Check only if ModelAdmin.get_list_display() isn't overridden.
        elif obj.get_list_display.__func__ is ModelAdmin.get_list_display:
            return list(chain.from_iterable(
                self._check_list_display_links_item(obj, field_name, "list_display_links[%d]" % index)
                for index, field_name in enumerate(obj.list_display_links)
            ))
        return []

    def _check_list_display_links_item(self, obj, field_name, label):
        if field_name not in obj.list_display:
            return [
                checks.Error(
                    "The value of '%s' refers to '%s', which is not defined in 'list_display'." % (
                        label, field_name
                    ),
                    obj=obj.__class__,
                    id='admin.E111',
                )
            ]
        else:
            return []

    def _check_list_filter(self, obj):
        if not isinstance(obj.list_filter, (list, tuple)):
            return must_be('a list or tuple', option='list_filter', obj=obj, id='admin.E112')
        else:
            return list(chain.from_iterable(
                self._check_list_filter_item(obj, item, "list_filter[%d]" % index)
                for index, item in enumerate(obj.list_filter)
            ))

    def _check_list_filter_item(self, obj, item, label):
        """
