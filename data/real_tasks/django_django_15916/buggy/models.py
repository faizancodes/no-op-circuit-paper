            kwargs["localize"] = True
        if labels and f.name in labels:
            kwargs["label"] = labels[f.name]
        if help_texts and f.name in help_texts:
            kwargs["help_text"] = help_texts[f.name]
        if error_messages and f.name in error_messages:
            kwargs["error_messages"] = error_messages[f.name]
        if field_classes and f.name in field_classes:
            kwargs["form_class"] = field_classes[f.name]

        if formfield_callback is None:
            formfield = f.formfield(**kwargs)
        elif not callable(formfield_callback):
            raise TypeError("formfield_callback must be a function or callable")
        else:
            formfield = formfield_callback(f, **kwargs)

        if formfield:
            if apply_limit_choices_to:
                apply_limit_choices_to_to_formfield(formfield)
            field_dict[f.name] = formfield
        else:
            ignored.append(f.name)
    if fields:
        field_dict = {
            f: field_dict.get(f)
            for f in fields
            if (not exclude or f not in exclude) and f not in ignored
        }
    return field_dict


class ModelFormOptions:
    def __init__(self, options=None):
        self.model = getattr(options, "model", None)
        self.fields = getattr(options, "fields", None)
        self.exclude = getattr(options, "exclude", None)
        self.widgets = getattr(options, "widgets", None)
        self.localized_fields = getattr(options, "localized_fields", None)
        self.labels = getattr(options, "labels", None)
        self.help_texts = getattr(options, "help_texts", None)
        self.error_messages = getattr(options, "error_messages", None)
        self.field_classes = getattr(options, "field_classes", None)


class ModelFormMetaclass(DeclarativeFieldsMetaclass):
    def __new__(mcs, name, bases, attrs):
        base_formfield_callback = None
        for b in bases:
            if hasattr(b, "Meta") and hasattr(b.Meta, "formfield_callback"):
                base_formfield_callback = b.Meta.formfield_callback
                break

        formfield_callback = attrs.pop("formfield_callback", base_formfield_callback)

        new_class = super().__new__(mcs, name, bases, attrs)

        if bases == (BaseModelForm,):
            return new_class

        opts = new_class._meta = ModelFormOptions(getattr(new_class, "Meta", None))

        # We check if a string was passed to `fields` or `exclude`,
        # which is likely to be a mistake where the user typed ('foo') instead
        # of ('foo',)
        for opt in ["fields", "exclude", "localized_fields"]:
            value = getattr(opts, opt)
            if isinstance(value, str) and value != ALL_FIELDS:
                msg = (
                    "%(model)s.Meta.%(opt)s cannot be a string. "
                    "Did you mean to type: ('%(value)s',)?"
                    % {
                        "model": new_class.__name__,
                        "opt": opt,
                        "value": value,
                    }
                )
                raise TypeError(msg)

        if opts.model:
            # If a model is defined, extract form fields from it.
            if opts.fields is None and opts.exclude is None:
                raise ImproperlyConfigured(
                    "Creating a ModelForm without either the 'fields' attribute "
                    "or the 'exclude' attribute is prohibited; form %s "
                    "needs updating." % name
                )

            if opts.fields == ALL_FIELDS:
                # Sentinel for fields_for_model to indicate "get the list of
                # fields from the model"
                opts.fields = None

            fields = fields_for_model(
                opts.model,
                opts.fields,
                opts.exclude,
                opts.widgets,
