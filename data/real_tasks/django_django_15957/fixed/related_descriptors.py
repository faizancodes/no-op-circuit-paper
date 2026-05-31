   ``ReverseOneToOneDescriptor``.

   One-to-one relations are asymmetrical, despite the apparent symmetry of the
   name, because they're implemented in the database with a foreign key from
   one table to another. As a consequence ``ReverseOneToOneDescriptor`` is
   slightly different from ``ForwardManyToOneDescriptor``.

4. Related objects manager for related instances on the reverse side of a
   many-to-one relation: ``ReverseManyToOneDescriptor``.

   Unlike the previous two classes, this one provides access to a collection
   of objects. It returns a manager rather than an instance.

5. Related objects manager for related instances on the forward or reverse
   sides of a many-to-many relation: ``ManyToManyDescriptor``.

   Many-to-many relations are symmetrical. The syntax of Django models
   requires declaring them on one side but that's an implementation detail.
   They could be declared on the other side without any change in behavior.
   Therefore the forward and reverse descriptors can be the same.

   If you're looking for ``ForwardManyToManyDescriptor`` or
   ``ReverseManyToManyDescriptor``, use ``ManyToManyDescriptor`` instead.
"""

from django.core.exceptions import FieldError
from django.db import connections, router, transaction
from django.db.models import Q, signals
from django.db.models.query import QuerySet
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import resolve_callables
from django.utils.functional import cached_property


class ForeignKeyDeferredAttribute(DeferredAttribute):
    def __set__(self, instance, value):
        if instance.__dict__.get(self.field.attname) != value and self.field.is_cached(
            instance
        ):
            self.field.delete_cached_value(instance)
        instance.__dict__[self.field.attname] = value


def _filter_prefetch_queryset(queryset, field_name, instances):
    predicate = Q(**{f"{field_name}__in": instances})
    if queryset.query.is_sliced:
        low_mark, high_mark = queryset.query.low_mark, queryset.query.high_mark
        order_by = [
            expr
            for expr, _ in queryset.query.get_compiler(
                using=queryset._db or DEFAULT_DB_ALIAS
            ).get_order_by()
        ]
        window = Window(RowNumber(), partition_by=field_name, order_by=order_by)
        predicate &= GreaterThan(window, low_mark)
        if high_mark is not None:
            predicate &= LessThanOrEqual(window, high_mark)
        queryset.query.clear_limits()
    return queryset.filter(predicate)


class ForwardManyToOneDescriptor:
    """
    Accessor to the related object on the forward side of a many-to-one or
    one-to-one (via ForwardOneToOneDescriptor subclass) relation.

    In the example::

        class Child(Model):
            parent = ForeignKey(Parent, related_name='children')

    ``Child.parent`` is a ``ForwardManyToOneDescriptor`` instance.
    """

    def __init__(self, field_with_rel):
        self.field = field_with_rel

    @cached_property
    def RelatedObjectDoesNotExist(self):
        # The exception can't be created at initialization time since the
        # related model might not be resolved yet; `self.field.model` might
        # still be a string model reference.
        return type(
            "RelatedObjectDoesNotExist",
            (self.field.remote_field.model.DoesNotExist, AttributeError),
            {
                "__module__": self.field.model.__module__,
                "__qualname__": "%s.%s.RelatedObjectDoesNotExist"
                % (
                    self.field.model.__qualname__,
                    self.field.name,
                ),
            },
        )

    def is_cached(self, instance):
        return self.field.is_cached(instance)

    def get_queryset(self, **hints):
        return self.field.remote_field.model._base_manager.db_manager(hints=hints).all()

    def get_prefetch_queryset(self, instances, queryset=None):
        if queryset is None:
            queryset = self.get_queryset()
