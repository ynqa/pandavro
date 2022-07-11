from collections import OrderedDict
from pathlib import Path
from typing import Optional, Iterable

import fastavro
import numpy as np
import pandas as pd
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
    np.dtype('O'): 'complex',  # FIXME: Don't automatically store objects as strings
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

# Int8 and Int16 don't exist in Pandas NA-dtypes
AVRO_TO_PANDAS_TYPES['int'] = pd.Int32Dtype
AVRO_TO_PANDAS_TYPES['long'] = pd.Int64Dtype
AVRO_TO_PANDAS_UNSIGNED_TYPES['int'] = pd.UInt32Dtype

# Recognize these Pandas dtypes
NUMPY_TO_AVRO_TYPES[pd.StringDtype()] = 'string'
NUMPY_TO_AVRO_TYPES[pd.BooleanDtype()] = 'boolean'

# Convert these to python first
PANDAS_TO_PYTHON_TYPES[np.bool_] = bool

# Indicate the optional return datatype
AVRO_TO_PANDAS_TYPES['string'] = pd.StringDtype
AVRO_TO_PANDAS_TYPES['boolean'] = pd.BooleanDtype


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


def __complex_field_infer(df, field, nested_record_names):
    NoneType = type(None)
    bool_types = {bool, NoneType}
    string_types = {str, NoneType}
    byte_types = {bytes, NoneType}
    record_types = {dict, OrderedDict, NoneType}
    array_types = {list, NoneType}

    base_field_types = set(df[field].apply(type))

    # String type - have to check for string first, in case a column contains
    # entirely 'None's
    if base_field_types.issubset(string_types):
        return 'string'
    # Bool type - if a boolean field contains missing values, pandas will give
    # its type as np.dtype('O'), so we have to double check for it here.
    if base_field_types.issubset(bool_types):
        return 'boolean'
    # Bytes type - have to check for bytes first, in case a column contains
    # entirely 'None's
    if base_field_types.issubset(byte_types):
        return 'bytes'
    # Record type
    elif base_field_types.issubset(record_types):
        records = df.loc[~df[field].isna(), field].reset_index(drop=True)

        if field in nested_record_names:
            nested_record_names[field] += 1
        else:
            nested_record_names[field] = 0
        return {
            'type': 'record',
            'name': field + '_record' + str(nested_record_names[field]),
            'fields': __fields_infer(pd.DataFrame.from_records(records),
                                     nested_record_names)
        }
    # Array type
    elif base_field_types.issubset(array_types):
        arrays = pd.Series(df.loc[~df[field].isna(), field].sum(),
                           name=field).reset_index(drop=True)
        if arrays.empty:
            print('Array field \'{}\' has been provided containing only empty '
                  'lists. The intended type of its contents cannot be '
                  'inferred, so \'string\' was assumed.'.format(field))
            items = 'string'
        else:
            items = __fields_infer(arrays.to_frame(),
                                   nested_record_names)[0]['type']
        return {
            'type': 'array',
            'items': items
        }


def __fields_infer(df, nested_record_names):
    inferred_fields = [
        {'name': key, 'type': __type_infer(type_np)}
        for key, type_np in df.dtypes.items()
    ]
    for field in inferred_fields:
        if 'complex' in field['type']:
            field['type'] = [
                'null',
                __complex_field_infer(df, field['name'], nested_record_names)
            ]
    return inferred_fields


def __convert_field_micros_to_millis(field):
    if isinstance(field, list):
        for i in range(0, len(field)):
            field[i] = __convert_field_micros_to_millis(field[i])
        return field
    elif isinstance(field, dict):
        for key, item in field.items():
            field[key] = __convert_field_micros_to_millis(item)
        return field
    elif isinstance(field, str):
        if field == 'timestamp-micros':
            return 'timestamp-millis'
        else:
            return field


def schema_infer(df, times_as_micros=True):
    """
    Infers the Avro schema of a pandas DataFrame

    Args:
        df: DataFrame to infer the schema of
        times_as_micros:
            Whether timestamps should be stored as microseconds (default)
            or milliseconds (as expected by Apache Hive)
    """
    fields = __fields_infer(df, {})
    schema = {
        'type': 'record',
        'name': 'Root',
        'fields': fields
    }

    # Patch 'timestamp-millis' in
    if not times_as_micros:
        for field in schema['fields']:
            field = __convert_field_micros_to_millis(field)
    return schema


def __file_to_dataframe(f, schema, na_dtypes=False, columns: Optional[Iterable[str]] = None, **kwargs):
    reader = fastavro.reader(f, reader_schema=schema)
    if columns is None:
        records = list(reader)
    # To free up some RAM we can select a subset of columns
    else:
        columns_set = frozenset(columns)
        records = [{k: v for k, v in row.items() if k in columns_set} for row in reader]

    df = pd.DataFrame.from_records(records, columns=columns, **kwargs)

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


def read_avro(file_path_or_buffer, schema=None, na_dtypes=False, columns: Optional[Iterable[str]] = None, **kwargs):
    """
    Avro file reader.

    Args:
        file_path_or_buffer: Input file path (str or pathlib.Path) or file-like object.
        schema: Avro schema.
        na_dtypes: Read int, long, string, boolean types back as Pandas NA-supporting datatypes.
        columns: Sequence, subset of columns to load in memory.
        **kwargs: Keyword argument to pandas.DataFrame.from_records.

    Returns:
        Class of pd.DataFrame.
    """
    if isinstance(file_path_or_buffer, Path):
        if not file_path_or_buffer.exists():
            raise FileExistsError
        file_path_or_buffer = str(file_path_or_buffer)

    if isinstance(file_path_or_buffer, str):
        with open(file_path_or_buffer, 'rb') as f:
            return __file_to_dataframe(f, schema, na_dtypes=na_dtypes, columns=columns, **kwargs)
    else:
        return __file_to_dataframe(
            file_path_or_buffer, schema, na_dtypes=na_dtypes, columns=columns, **kwargs
        )


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


def __preprocess_dicts(l):
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
            Output file path (str or pathlib.Path) or file-like object.
        df: pd.DataFrame.
        schema: Dict of Avro schema.
            If it's set None, inferring schema.
        append: Boolean to control if will append to existing file
        times_as_micros: If True (default), save datetimes with microseconds resolution. If False, save with millisecond
            resolution instead.
        kwargs: Keyword arguments to fastavro.writer

    """
    if schema is None:
        schema = schema_infer(df, times_as_micros)

    open_mode = 'wb' if not append else 'a+b'

    records = __preprocess_dicts(df.to_dict('records'))

    if isinstance(file_path_or_buffer, Path):
        file_path_or_buffer = str(file_path_or_buffer)

    if isinstance(file_path_or_buffer, str):
        with open(file_path_or_buffer, open_mode) as f:
            fastavro.writer(f, schema=schema,
                            records=records, **kwargs)
    else:
        fastavro.writer(file_path_or_buffer, schema=schema,
                        records=records, **kwargs)
