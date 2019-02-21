import fastavro
import numpy as np
import pandas as pd
import six

# Pandas 0.24 added support for nullable integers. Include those in the supported
# integer dtypes if present, otherwise ignore them.
try:
    from pandas import (
        Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype,
        UInt8Dtype, UInt16Dtype, UInt32Dtype, UInt64Dtype
    )

    _PANDAS_INTEGER_DTYPES = (
        Int8Dtype, Int16Dtype, Int32Dtype, UInt8Dtype, UInt16Dtype, UInt32Dtype
    )
    _PANDAS_LONG_DTYPES = (Int64Dtype, UInt64Dtype)
except ImportError:
    _PANDAS_INTEGER_DTYPES = ()
    _PANDAS_LONG_DTYPES = ()



def __type_infer(t):
    if t is np.bool_:
        return 'boolean'
    elif t in (np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32) + _PANDAS_INTEGER_DTYPES:
        return 'int'
    elif t in (np.int64, np.uint64) + _PANDAS_LONG_DTYPES:
        return 'long'
    elif t is np.float32:
        return 'float'
    elif t is np.float64:
        return 'double'
    elif t in (np.object_, np.unicode_):
        # TODO: Dealing with the case of collection.
        return 'string'
    elif t.type in (np.datetime64, pd.core.dtypes.dtypes.DatetimeTZDtypeType):
        # https://avro.apache.org/docs/current/spec.html#Timestamp+%28microsecond+precision%29)
        return {'type': 'long', 'logicalType': 'timestamp-micros'}
    else:
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
