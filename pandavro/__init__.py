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


DTYPE_TO_AVRO_TYPE = {
    np.bool_: 'boolean',
    np.int8: 'int',
    np.int16: 'int',
    np.int32: 'int',
    np.uint8: 'int',
    np.uint16: 'int',
    np.uint32: 'int',
    np.int64: 'long',
    np.uint64: 'long',
    np.object_: 'string',
    np.unicode_: 'string',
    np.float32: 'float',
    np.float64: 'double',
    np.datetime64: {'type': 'long', 'logicalType': 'timestamp-micros'},
    DatetimeTZDtype: {'type': 'long', 'logicalType': 'timestamp-micros'},
    np.void: 'binary',
    np.bytes_: 'binary',
}


# Pandas 0.24 added support for nullable integers. Include those in the supported
# integer dtypes if present, otherwise ignore them.
try:
    DTYPE_TO_AVRO_TYPE[pd.Int8Dtype] = 'int'
    DTYPE_TO_AVRO_TYPE[pd.Int16Dtype] = 'int'
    DTYPE_TO_AVRO_TYPE[pd.Int32Dtype] = 'int'
    DTYPE_TO_AVRO_TYPE[pd.UInt8Dtype] = 'int'
    DTYPE_TO_AVRO_TYPE[pd.UInt16Dtype] = 'int'
    DTYPE_TO_AVRO_TYPE[pd.UInt32Dtype] = 'int'
    DTYPE_TO_AVRO_TYPE[pd.UInt64Dtype] = 'long'
    DTYPE_TO_AVRO_TYPE[pd.Int64Dtype] = 'long'
except AttributeError:
    pass


def __type_infer(t):
    if t in DTYPE_TO_AVRO_TYPE:
        return DTYPE_TO_AVRO_TYPE[t]
    elif getattr(t, 'type', None) in DTYPE_TO_AVRO_TYPE:
        return DTYPE_TO_AVRO_TYPE[t.type]
    raise TypeError('Invalid type: {}'.format(t))


def __fields_infer(df):
    fields = []
    for key, type_np in six.iteritems(df.dtypes):
        type_avro = __type_infer(type_np)
        fields.append({'name': key, 'type': ['null', type_avro]})
    return fields


def __schema_infer(df):
    fields = __fields_infer(df)
    schema = {
        'type': 'record',
        'name': 'Root',
        'fields': fields
    }
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


def to_avro(file_path, df, schema=None):
    """
    Avro file writer.

    Args:
        file_path: Output file path.
        df: pd.DataFrame.
        schema: Dict of Avro schema.
            If it's set None, inferring schema.
    """

    if schema is None:
        schema = __schema_infer(df)

    with open(file_path, 'wb') as f:
        fastavro.writer(f, schema=schema,
                        records=df.to_dict('records'))
