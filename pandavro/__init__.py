import fastavro
import numpy as np
import pandas as pd


def __type_infer(t: np.dtype):
    if t == np.bool_:
        return 'boolean'
    elif t == (np.int8 or np.int16 or np.int32):
        return 'int'
    elif t == np.int64:
        return 'long'
    elif t == np.float32:
        return 'float'
    elif t == np.float64:
        return 'double'
    elif t == np.object:
        # TODO: Dealing with the case of collection.
        return 'string'
    else:
        raise TypeError('Invalid type: {}'.format(t))


def __fields_infer(df: pd.DataFrame):
    fields = []
    for key, type_np in df.dtypes.iteritems():
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


def from_avro(file_path, schema=None):
    """
    Avro file reader.

    Args:
        file_path: Input file path.
        schema: Avro schema.

    Returns:
        Class of pd.DataFrame.
    """

    with open(file_path, 'rb') as f:
        reader = fastavro.reader(f, reader_schema=schema)
        return pd.DataFrame.from_records(list(reader))


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
