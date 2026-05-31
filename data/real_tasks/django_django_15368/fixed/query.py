"""
The main QuerySet implementation. This provides the public API for the ORM.
"""

import copy
import operator
import warnings
from itertools import chain, islice

import django
from django.conf import settings
from django.core import exceptions
from django.db import (
    DJANGO_VERSION_PICKLE_KEY, IntegrityError, NotSupportedError, connections,
    router, transaction,
)
from django.db.models import AutoField, DateField, DateTimeField, sql
from django.db.models.constants import LOOKUP_SEP, OnConflict
from django.db.models.deletion import Collector
from django.db.models.expressions import Case, F, Ref, Value, When
from django.db.models.functions import Cast, Trunc
from django.db.models.query_utils import FilteredRelation, Q
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE
from django.db.models.utils import create_namedtuple_class, resolve_callables
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango50Warning
from django.utils.functional import cached_property, partition

# The maximum number of results to fetch in a get() query.
MAX_GET_RESULTS = 21

# The maximum number of items to display in a QuerySet.__repr__
REPR_OUTPUT_SIZE = 20


class BaseIterable:
    def __init__(self, queryset, chunked_fetch=False, chunk_size=GET_ITERATOR_CHUNK_SIZE):
        self.queryset = queryset
        self.chunked_fetch = chunked_fetch
        self.chunk_size = chunk_size


class ModelIterable(BaseIterable):
    """Iterable that yields a model instance for each row."""

    def __iter__(self):
        queryset = self.queryset
        db = queryset.db
        compiler = queryset.query.get_compiler(using=db)
        # Execute the query. This will also fill compiler.select, klass_info,
        # and annotations.
        results = compiler.execute_sql(chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size)
        select, klass_info, annotation_col_map = (compiler.select, compiler.klass_info,
                                                  compiler.annotation_col_map)
        model_cls = klass_info['model']
        select_fields = klass_info['select_fields']
        model_fields_start, model_fields_end = select_fields[0], select_fields[-1] + 1
        init_list = [f[0].target.attname
                     for f in select[model_fields_start:model_fields_end]]
        related_populators = get_related_populators(klass_info, select, db)
        known_related_objects = [
            (field, related_objs, operator.attrgetter(*[
                field.attname
