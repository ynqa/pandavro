import logging

import fastavro
import numpy as np
import pandas as pd
import six

logger = logging.getLogger(__name__)

try:
    # Pandas <= 0.23
    from pandas.core.dtypes.dtypes import DatetimeTZDtypeType as DatetimeTZDtype
except ImportError:
    # Pandas >= 0.24
    from pandas import DatetimeTZDtype

NUMPY_TO_AVRO_TYPES = {
    np.dtype('?'): 'boolean',
    # pd.[U]Int[6/16/32/64]Dtype() is covered by these numpy types
    np.int8: 'int',
    np.int16: 'int',
    np.int32: 'int',
    np.uint8: {'type': 'int', 'unsigned': True},
    np.uint16: {'type': 'int', 'unsigned': True},
    np.uint32: {'type': 'int', 'unsigned': True},
    np.int64: 'long',
    np.uint64: {'type': 'long', 'unsigned': True},
    np.dtype('O'): 'string',  # FIXME: Don't automatically store objects as strings
    np.unicode_: 'string',
    np.float32: 'float',
    np.float64: 'double',
    np.datetime64: {'type': 'long', 'logicalType': 'timestamp-micros'},
    DatetimeTZDtype: {'type': 'long', 'logicalType': 'timestamp-micros'},
    pd.Timestamp: {'type': 'long', 'logicalType': 'timestamp-micros'},
}

# This is used for forced conversion to Pandas NA-dtypes
AVRO_TO_PANDAS_TYPES = {}
# We use this extra dict for unsigned ints, adding it allows the dicts to stay a nice simple mapping
AVRO_TO_PANDAS_UNSIGNED_TYPES = {}

# This is used to convert Pandas NA-dtypes to python so fastavro can write
PANDAS_TO_PYTHON_TYPES = {}

# Pandas 0.24 added support for nullable integers. Include those in the supported
# integer dtypes if present, otherwise ignore them.
try:
    # Int8 and Int16 don't exist in Pandas NA-dtypes
    AVRO_TO_PANDAS_TYPES['int'] = pd.Int32Dtype
    AVRO_TO_PANDAS_TYPES['long'] = pd.Int64Dtype
    AVRO_TO_PANDAS_UNSIGNED_TYPES['int'] = pd.UInt32Dtype

    logger.debug("Imported pandas >=0.24 integer datatypes")
except AttributeError:
    logger.debug("Did not import pandas >=0.24 integer datatypes")

try:
    # Recognize these Pandas dtypes
    NUMPY_TO_AVRO_TYPES[pd.StringDtype()] = 'string'
    NUMPY_TO_AVRO_TYPES[pd.BooleanDtype()] = 'boolean'

    # Convert these to python first
    PANDAS_TO_PYTHON_TYPES[np.bool_] = bool

    # Indicate the optional return datatype
    AVRO_TO_PANDAS_TYPES['string'] = pd.StringDtype
    AVRO_TO_PANDAS_TYPES['boolean'] = pd.BooleanDtype

    logger.debug("Imported pandas >=1.0.0 datatypes")
except AttributeError:
    logger.debug("Did not import pandas >=1.0.0 datatypes")


def __type_infer(t):
    # Binary data has to be handled separately from the other dtypes because it
    # requires a parameter, the buffer size.
    print(t)
    if t is np.void:
        return {
            'type': ['null', 'fixed'],
            'size': t.itemsize,
        }

    if t in NUMPY_TO_AVRO_TYPES:
        avro_type = NUMPY_TO_AVRO_TYPES[t]
        if isinstance(avro_type, dict):
            # To ensure that the global is unmodified if millis are inserted
            avro_type = avro_type.copy()
        return ['null', avro_type]
    if hasattr(t, 'type'):
        return __type_infer(t.type)

    raise TypeError('Invalid type: {}'.format(t))


def __fields_infer(df):
    return [
        {'name': key, 'type': __type_infer(type_np)}
        for key, type_np in six.iteritems(df.dtypes)
    ]


def __schema_infer(df, times_as_micros):
    fields = __fields_infer(df)
    schema = {
        'type': 'record',
        'name': 'Root',
        'fields': fields
    }

    # Patch 'timestamp-millis' in
    if not times_as_micros:
        for field in schema['fields']:
            non_null_type = field['type'][1]
            if isinstance(non_null_type, dict):
                if non_null_type.get('logicalType') == 'timestamp-micros':
                    non_null_type['logicalType'] = 'timestamp-millis'
    return schema


def __file_to_dataframe(f, schema, na_dtypes=False, **kwargs):
    reader = fastavro.reader(f, reader_schema=schema)
    df = pd.DataFrame.from_records(list(reader), **kwargs)

    def _filter(typelist):
        # It's a string, we return it directly
        if type(typelist) == str:
            return typelist, False
        # If a logical type dict, it has a type attribute
        elif type(typelist) == dict:
            # Return None as we don't touch logical types
            if typelist.get('logicalType'):
                return None, False
            elif typelist.get('unsigned'):
                return typelist['type'], True
        # It's a list and we filter any "null"
        else:
            l = [t for t in typelist if t != "null"]
            if len(l) > 1:
                raise ValueError("More items in Avro schema type list than 1: '{d}'".format(l))
            return _filter(l[0])

    if na_dtypes:
        # Look at schema here, map Avro types to available Pandas 1.0 dtypes
        # Then convert dtypes in place to these new dtypes in a deterministic way
        # We know this is possible as we know the Avro type
        for field in reader.writer_schema["fields"]:
            t, u = _filter(field["type"])
            name = field["name"]
            if name in df.columns:
                if not u:
                    if t in AVRO_TO_PANDAS_TYPES:
                        df[name] = df[name].astype(AVRO_TO_PANDAS_TYPES[t]())
                else:
                    if t in AVRO_TO_PANDAS_UNSIGNED_TYPES:
                        df[name] = df[name].astype(AVRO_TO_PANDAS_UNSIGNED_TYPES[t]())
    return df


def read_avro(file_path_or_buffer, schema=None, na_dtypes=False, **kwargs):
    """
    Avro file reader.

    Args:
        file_path_or_buffer: Input file path or file-like object.
        schema: Avro schema.
        na_dtypes: Read int, long, string, boolean types back as Pandas NA-supporting datatypes.
        **kwargs: Keyword argument to pandas.DataFrame.from_records.

    Returns:
        Class of pd.DataFrame.
    """
    if isinstance(file_path_or_buffer, six.string_types):
        with open(file_path_or_buffer, 'rb') as f:
            return __file_to_dataframe(f, schema, na_dtypes=na_dtypes, **kwargs)
    else:
        return __file_to_dataframe(file_path_or_buffer, schema, na_dtypes=na_dtypes, **kwargs)


def from_avro(file_path_or_buffer, schema=None, na_dtypes=False, **kwargs):
    """
    Avro file reader.

    Delegates to the `read_avro` method to remain backward compatible.

    Args:
        file_path_or_buffer: Input file path or file-like object.
        schema: Avro schema.
        na_dtypes: Read int, long, string, boolean types back as Pandas NA-supporting datatypes.
        **kwargs: Keyword argument to pandas.DataFrame.from_records.

    Returns:
        Class of pd.DataFrame.
    """
    return read_avro(file_path_or_buffer, schema, na_dtypes=na_dtypes, **kwargs)


def _preprocess_dicts(l):
    "Preprocess list of dicts inplace for fastavro compatibility"
    for d in l:
        for k, v in d.items():
            # Replace pd.NA with None so fastavro can write it
            if v is pd.NA:
                d[k] = None
            # Convert some Pandas dtypes to normal Python dtypes
            for key, value in PANDAS_TO_PYTHON_TYPES.items():
                if isinstance(v, key):
                    d[k] = value(v)
    return l


def to_avro(file_path_or_buffer, df, schema=None, append=False,
            times_as_micros=True, **kwargs):
    """
    Avro file writer.

    Args:
        file_path_or_buffer:
            Output file path or file-like object.
        df: pd.DataFrame.
        schema: Dict of Avro schema.
            If it's set None, inferring schema.
        append: Boolean to control if will append to existing file
        times_as_micros: If True (default), save datetimes with microseconds resolution. If False, save with millisecond
            resolution instead.
        kwargs: Keyword arguments to fastavro.writer

    """
    if schema is None:
        schema = __schema_infer(df, times_as_micros)

    open_mode = 'wb' if not append else 'a+b'

    # This special kwarg is only to enable testing performance difference
    if kwargs.get("_test_preprocess_off"):
        kwargs.pop("_test_preprocess_off")
        records = df.to_dict('records')
    else:
        records = _preprocess_dicts(df.to_dict('records'))

    if isinstance(file_path_or_buffer, six.string_types):
        with open(file_path_or_buffer, open_mode) as f:
            fastavro.writer(f, schema=schema,
                            records=records, **kwargs)
    else:
        fastavro.writer(file_path_or_buffer, schema=schema,
                        records=records, **kwargs)
