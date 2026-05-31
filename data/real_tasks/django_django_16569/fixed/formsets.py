                    self.error_messages["too_few_forms"] % {"num": self.min_num},
                    code="too_few_forms",
                )
            # Give self.clean() a chance to do cross-form validation.
            self.clean()
        except ValidationError as e:
            self._non_form_errors = self.error_class(
                e.error_list,
                error_class="nonform",
                renderer=self.renderer,
            )

    def clean(self):
        """
        Hook for doing any extra formset-wide cleaning after Form.clean() has
        been called on every form. Any ValidationError raised by this method
        will not be associated with a particular form; it will be accessible
        via formset.non_form_errors()
        """
        pass

    def has_changed(self):
        """Return True if data in any form differs from initial."""
        return any(form.has_changed() for form in self)

    def add_fields(self, form, index):
        """A hook for adding extra fields on to each form instance."""
        initial_form_count = self.initial_form_count()
        if self.can_order:
            # Only pre-fill the ordering field for initial forms.
            if index is not None and index < initial_form_count:
                form.fields[ORDERING_FIELD_NAME] = IntegerField(
                    label=_("Order"),
                    initial=index + 1,
                    required=False,
                    widget=self.get_ordering_widget(),
                )
            else:
                form.fields[ORDERING_FIELD_NAME] = IntegerField(
                    label=_("Order"),
                    required=False,
                    widget=self.get_ordering_widget(),
                )
        if self.can_delete and (
            self.can_delete_extra or (index is not None and index < initial_form_count)
        ):
            form.fields[DELETION_FIELD_NAME] = BooleanField(
                label=_("Delete"),
                required=False,
                widget=self.get_deletion_widget(),
            )

    def add_prefix(self, index):
        return "%s-%s" % (self.prefix, index)

    def is_multipart(self):
        """
        Return True if the formset needs to be multipart, i.e. it
        has FileInput, or False otherwise.
        """
        if self.forms:
            return self.forms[0].is_multipart()
        else:
            return self.empty_form.is_multipart()

    @property
    def media(self):
        # All the forms on a FormSet are the same, so you only need to
        # interrogate the first form for media.
        if self.forms:
            return self.forms[0].media
        else:
            return self.empty_form.media

    @property
    def template_name(self):
        return self.renderer.formset_template_name

    def get_context(self):
        return {"formset": self}


def formset_factory(
    form,
    formset=BaseFormSet,
    extra=1,
    can_order=False,
    can_delete=False,
    max_num=None,
