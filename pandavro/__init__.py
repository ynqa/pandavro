import fastavro
import numpy as np
import pandas as pd
import six

try:
    # Pandas <= 0.23
    from pandas.core.dtypes.dtypes import DatetimeTZDtypeType as DatetimeTZDtype
except ImportError:
    # Pandas >= 0.24
    from pandas import DatetimeTZDtype

NUMPY_TO_AVRO_TYPES = {
    np.dtype('?'): 'boolean',
    np.int8: 'int',
    np.int16: 'int',
    np.int32: 'int',
    np.uint8: 'int',
    np.uint16: 'int',
    np.uint32: 'int',
    np.int64: 'long',
    np.uint64: 'long',
    np.dtype('O'): 'string',  # FIXME: Don't automatically store objects as strings
    np.unicode_: 'string',
    np.float32: 'float',
    np.float64: 'double',
    np.datetime64: {'type': 'long', 'logicalType': 'timestamp-micros'},
    DatetimeTZDtype: {'type': 'long', 'logicalType': 'timestamp-micros'},
    pd.Timestamp: {'type': 'long', 'logicalType': 'timestamp-micros'},
}

# Pandas 0.24 added support for nullable integers. Include those in the supported
# integer dtypes if present, otherwise ignore them.
try:
    NUMPY_TO_AVRO_TYPES[pd.Int8Dtype] = 'int'
    NUMPY_TO_AVRO_TYPES[pd.Int16Dtype] = 'int'
    NUMPY_TO_AVRO_TYPES[pd.Int32Dtype] = 'int'
    NUMPY_TO_AVRO_TYPES[pd.Int64Dtype] = 'long'

    # We need the non-standard `unsigned` flag because Avro doesn't support
    # unsigned integers, and we have no other way of indicating that the loaded
    # integer is supposed to be unsigned.
    NUMPY_TO_AVRO_TYPES[pd.UInt8Dtype] = {'type': 'int', 'unsigned': True}
    NUMPY_TO_AVRO_TYPES[pd.UInt16Dtype] = {'type': 'int', 'unsigned': True}
    NUMPY_TO_AVRO_TYPES[pd.UInt32Dtype] = {'type': 'int', 'unsigned': True}
    NUMPY_TO_AVRO_TYPES[pd.UInt64Dtype] = {'type': 'long', 'unsigned': True}
except AttributeError:
    pass


def __type_infer(t):
    # Binary data has to be handled separately from the other dtypes because it
    # requires a parameter, the buffer size.
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


def __file_to_dataframe(f, schema, **kwargs):
    reader = fastavro.reader(f, reader_schema=schema)
    return pd.DataFrame.from_records(list(reader), **kwargs)


def read_avro(file_path_or_buffer, schema=None, **kwargs):
    """
    Avro file reader.

    Args:
        file_path_or_buffer: Input file path or file-like object.
        schema: Avro schema.
        **kwargs: Keyword argument to pandas.DataFrame.from_records.

    Returns:
        Class of pd.DataFrame.
    """
    if isinstance(file_path_or_buffer, six.string_types):
        with open(file_path_or_buffer, 'rb') as f:
            return __file_to_dataframe(f, schema, **kwargs)
    else:
        return __file_to_dataframe(file_path_or_buffer, schema, **kwargs)


def from_avro(file_path_or_buffer, schema=None, **kwargs):
    """
    Avro file reader.

    Delegates to the `read_avro` method to remain backward compatible.

    Args:
        file_path_or_buffer: Input file path or file-like object.
        schema: Avro schema.
        **kwargs: Keyword argument to pandas.DataFrame.from_records.

    Returns:
        Class of pd.DataFrame.
    """
    return read_avro(file_path_or_buffer, schema, **kwargs)


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
        kwargs: Keyword arguments to fastavro.writer

    """
    if schema is None:
        schema = __schema_infer(df, times_as_micros)

    open_mode = 'wb' if not append else 'a+b'

    if isinstance(file_path_or_buffer, six.string_types):
        with open(file_path_or_buffer, open_mode) as f:
            fastavro.writer(f, schema=schema,
                            records=df.to_dict('records'), **kwargs)
    else:
        fastavro.writer(file_path_or_buffer, schema=schema,
                        records=df.to_dict('records'), **kwargs)
