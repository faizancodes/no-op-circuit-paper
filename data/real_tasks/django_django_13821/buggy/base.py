from django.utils.asyncio import async_unsafe
from django.utils.dateparse import parse_datetime, parse_time
from django.utils.duration import duration_microseconds
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import PY38

from .client import DatabaseClient
from .creation import DatabaseCreation
from .features import DatabaseFeatures
from .introspection import DatabaseIntrospection
from .operations import DatabaseOperations
from .schema import DatabaseSchemaEditor


def decoder(conv_func):
    """
    Convert bytestrings from Python's sqlite3 interface to a regular string.
    """
    return lambda s: conv_func(s.decode())


def none_guard(func):
    """
    Decorator that returns None if any of the arguments to the decorated
    function are None. Many SQL functions return NULL if any of their arguments
    are NULL. This decorator simplifies the implementation of this for the
    custom functions registered below.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return None if None in args else func(*args, **kwargs)
    return wrapper


def list_aggregate(function):
    """
    Return an aggregate class that accumulates values in a list and applies
    the provided function to the data.
    """
    return type('ListAggregate', (list,), {'finalize': function, 'step': list.append})


def check_sqlite_version():
    if Database.sqlite_version_info < (3, 8, 3):
        raise ImproperlyConfigured('SQLite 3.8.3 or later is required (found %s).' % Database.sqlite_version)


check_sqlite_version()

Database.register_converter("bool", b'1'.__eq__)
Database.register_converter("time", decoder(parse_time))
Database.register_converter("datetime", decoder(parse_datetime))
Database.register_converter("timestamp", decoder(parse_datetime))

Database.register_adapter(decimal.Decimal, str)


class DatabaseWrapper(BaseDatabaseWrapper):
    vendor = 'sqlite'
    display_name = 'SQLite'
    # SQLite doesn't actually support most of these types, but it "does the right
    # thing" given more verbose field definitions, so leave them as is so that
    # schema inspection is more useful.
    data_types = {
        'AutoField': 'integer',
        'BigAutoField': 'integer',
        'BinaryField': 'BLOB',
        'BooleanField': 'bool',
        'CharField': 'varchar(%(max_length)s)',
        'DateField': 'date',
        'DateTimeField': 'datetime',
        'DecimalField': 'decimal',
        'DurationField': 'bigint',
        'FileField': 'varchar(%(max_length)s)',
        'FilePathField': 'varchar(%(max_length)s)',
        'FloatField': 'real',
        'IntegerField': 'integer',
        'BigIntegerField': 'bigint',
        'IPAddressField': 'char(15)',
        'GenericIPAddressField': 'char(39)',
        'JSONField': 'text',
        'NullBooleanField': 'bool',
        'OneToOneField': 'integer',
        'PositiveBigIntegerField': 'bigint unsigned',
        'PositiveIntegerField': 'integer unsigned',
        'PositiveSmallIntegerField': 'smallint unsigned',
        'SlugField': 'varchar(%(max_length)s)',
        'SmallAutoField': 'integer',
