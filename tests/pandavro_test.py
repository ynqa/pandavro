import subprocess
from datetime import timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

try:
    # pandas >2.0
    from pandas.testing import assert_frame_equal
except ImportError:
    # previous version of pandas
    from pandas.util.testing import assert_frame_equal

import pandavro as pdx


@pytest.fixture
def dataframe():
    strings = ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar']
    return pd.DataFrame({"Boolean": [True, False, True, False, True, False, True, False],
                         "DateTime64": pd.date_range('20190101', '20190108', freq="1D", tz=timezone.utc),
                         "Float64": np.random.randn(8),
                         "Int64": np.random.randint(0, 10, 8),
                         "String": strings,
                         "Bytes": [string.encode() for string in strings],
                         })


def process_datetime64_column(df):
    df['DateTime64'] = df['DateTime64'].apply(lambda t: t.tz_convert(timezone.utc))


def test_schema_infer(dataframe):
    expect = {
        'type': 'record',
        'name': 'Root',
        'fields':
            [
                {'type': ['null', 'boolean'], 'name': 'Boolean'},
                {'type': ['null', {'logicalType': 'timestamp-micros', 'type': 'long'}],
                    'name': 'DateTime64'},
                {'type': ['null', 'double'], 'name': 'Float64'},
                {'type': ['null', 'long'], 'name': 'Int64'},
                {'type': ['null', 'string'], 'name': 'String'},
                {'type': ['null', 'bytes'], 'name': 'Bytes'},
            ]
    }
    assert expect == pdx.schema_infer(dataframe, times_as_micros=True)


def test_schema_infer_times_as_millis(dataframe):
    expect = {
        'type': 'record',
        'name': 'Root',
        'fields':
            [
                {'type': ['null', 'boolean'], 'name': 'Boolean'},
                {'type': ['null', {'logicalType': 'timestamp-millis', 'type': 'long'}],
                    'name': 'DateTime64'},
                {'type': ['null', 'double'], 'name': 'Float64'},
                {'type': ['null', 'long'], 'name': 'Int64'},
                {'type': ['null', 'string'], 'name': 'String'},
                {'type': ['null', 'bytes'], 'name': 'Bytes'},
            ]
    }
    assert expect == pdx.schema_infer(dataframe, times_as_micros=False)


def test_schema_infer_complex_types(dataframe):
    expect = {
        'type': 'record',
        'name': 'Root',
        'fields':
            [
                {'type': ['null', 'boolean'], 'name': 'Boolean'},
                {'type': ['null', {'logicalType': 'timestamp-micros', 'type': 'long'}],
                    'name': 'DateTime64'},
                {'type': ['null', 'double'], 'name': 'Float64'},
                {'type': ['null', 'long'], 'name': 'Int64'},
                {'type': ['null', 'string'], 'name': 'String'},
                {'type': ['null', 'bytes'], 'name': 'Bytes'},
                {'type': ['null', {
                    'fields':
                        [
                            {'name': 'field1', 'type': ['null', 'long']},
                            {'name': 'field2', 'type': ['null', 'string']}
                        ],
                    'name': 'Record_record0',
                    'type': 'record'}],
                 'name': 'Record'},
                {'type': ['null', {'type': 'array', 'items': ['null', 'long']}],
                 'name': 'Array'}
            ]
    }
    dataframe["Record"] = [
        {'field1': 1, 'field2': 'str1'}, {'field1': 2, 'field2': 'str2'},
        {'field1': 3, 'field2': 'str3'}, {'field1': 4, 'field2': 'str4'},
        {'field1': 5, 'field2': 'str5'}, {'field1': 6, 'field2': 'str6'},
        {'field1': 7, 'field2': 'str7'}, {'field1': 8, 'field2': 'str8'}]
    dataframe["Array"] = [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]
    ]

    assert expect == pdx.schema_infer(dataframe, times_as_micros=True)


def test_fields_infer(dataframe):
    expect = [
        {'type': ['null', 'boolean'], 'name': 'Boolean'},
        {'type': ['null', {'logicalType': 'timestamp-micros', 'type': 'long'}],
            'name': 'DateTime64'},
        {'type': ['null', 'double'], 'name': 'Float64'},
        {'type': ['null', 'long'], 'name': 'Int64'},
        {'type': ['null', 'string'], 'name': 'String'},
        {'type': ['null', 'bytes'], 'name': 'Bytes'},
    ]
    assert expect == pdx.__fields_infer(dataframe, nested_record_names={})


def test_buffer_e2e(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    with open(tf.name, 'rb') as f:
        expect = pdx.read_avro(BytesIO(f.read()))
        process_datetime64_column(expect)
    assert_frame_equal(expect, dataframe)


def test_file_path_e2e(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    expect = pdx.read_avro(tf.name)
    process_datetime64_column(expect)
    assert_frame_equal(expect, dataframe)


def test_pathlib_e2e(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(Path(tf.name), dataframe)
    expect = pdx.read_avro(Path(tf.name))
    process_datetime64_column(expect)
    assert_frame_equal(expect, dataframe)


def test_delegation(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    expect = pdx.from_avro(tf.name)
    process_datetime64_column(expect)
    assert_frame_equal(expect, dataframe)


def test_append(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe[0:int(dataframe.shape[0] / 2)])
    pdx.to_avro(tf.name, dataframe[int(dataframe.shape[0] / 2):], append=True)
    expect = pdx.from_avro(tf.name)
    process_datetime64_column(expect)
    assert_frame_equal(expect, dataframe)


def test_dataframe_subset_columns(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    columns = ['Boolean', 'Int64', 'String']
    expect = pdx.read_avro(tf.name, columns=columns)
    df = dataframe[columns]
    assert_frame_equal(expect, df)


def test_dataframe_kwargs(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    # exclude columns
    columns = ['String', 'Boolean']
    expect = pdx.read_avro(tf.name, exclude=columns)
    process_datetime64_column(expect)
    df = dataframe.drop(columns, axis=1)
    assert_frame_equal(expect, df)
    # specify index
    index = 'String'
    expect = pdx.read_avro(tf.name, index=index)
    process_datetime64_column(expect)
    df = dataframe.set_index(index)
    assert_frame_equal(expect, df)
    # specify nrows + exclude columns
    columns = ['String', 'Boolean']
    expect = pdx.read_avro(tf.name, exclude=columns, nrows=3)
    process_datetime64_column(expect)
    df = dataframe.drop(columns, axis=1).head(3)
    assert_frame_equal(expect, df)


@pytest.fixture
def dataframe_na_dtypes():
    def randints(dtype, length=8, nones=2):
        "Make random ints with 'nones' NAs randomly placed"
        s = pd.Series(list(np.random.randint(0, 10, 8)), dtype=dtype)
        for i in np.random.choice(range(length), size=nones):
            s[i] = None
        return s

    return pd.DataFrame({
        "Boolean": [True, False, True, False, True, False, True, False],
        "pdBoolean": pd.Series([True, False, True, False, True, False, True, False], dtype=pd.BooleanDtype()),
        "DateTime64": pd.date_range('20190101', '20190108', freq="1D"),
        "Float64": np.random.randn(8),
        "String": ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
        "pdString": pd.Series(['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar'], dtype=pd.StringDtype()),
        "Int8": randints(dtype=np.int8, nones=0),
        "Int16": randints(dtype=np.int16, nones=0),
        "Int32": randints(dtype=np.int32, nones=0),
        "Int64": randints(dtype=np.int64, nones=0),
        "UInt8": randints(dtype=np.uint8, nones=0),
        "UInt16": randints(dtype=np.uint16, nones=0),
        "UInt32": randints(dtype=np.uint32, nones=0),
        "UInt64": randints(dtype=np.uint64, nones=0),
        "pdInt8": randints(dtype=pd.Int8Dtype()),
        "pdInt16": randints(dtype=pd.Int16Dtype()),
        "pdInt32": randints(dtype=pd.Int32Dtype()),
        "pdInt64": randints(dtype=pd.Int64Dtype()),
        "pdUInt8": randints(dtype=pd.UInt8Dtype()),
        "pdUInt16": randints(dtype=pd.UInt16Dtype()),
        "pdUInt32": randints(dtype=pd.UInt32Dtype()),
        # "pdUInt64": randints(dtype=pd.UInt64Dtype()),
    })


def test_advanced_dtypes(dataframe_na_dtypes):
    "Should be able to write and read Pandas 1.0 NaN-compatible dtypes"
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe_na_dtypes)

    # Bools and datetime
    columns = ['Boolean', 'pdBoolean']
    expect = pdx.read_avro(tf.name, columns=columns, na_dtypes=True)
    df = dataframe_na_dtypes[columns]
    # We load everything as NA-dtypes
    df["Boolean"] = df["Boolean"].astype(pd.BooleanDtype())
    assert_frame_equal(expect, df)

    # Floats and ints
    columns = ['Float64', 'Int64', 'pdInt64']
    expect = pdx.read_avro(tf.name, columns=columns, na_dtypes=True)
    df = dataframe_na_dtypes[columns]
    df["Int64"] = df["Int64"].astype(pd.Int64Dtype())
    assert_frame_equal(expect, df)

    # Strings
    columns = ['String', 'pdString']
    expect = pdx.read_avro(tf.name, columns=columns, na_dtypes=True)
    df = dataframe_na_dtypes[columns]
    df["String"] = df["String"].astype(pd.StringDtype())
    assert_frame_equal(expect, df)


def test_ints(dataframe_na_dtypes):
    "Should write and read (unsigned) ints"
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe_na_dtypes)

    print(subprocess.run(["fastavro", "--schema", tf.name]))

    # Numpy Ints
    # FIXME: fastavro reads all ints as np.int64
    columns = ['Int8', 'Int16', 'Int32', 'Int64']
    expect = pdx.read_avro(tf.name, columns=columns, na_dtypes=False)
    df = dataframe_na_dtypes[columns]
    # Avro does not have the concept of 8 or 16-bit int
    df["Int8"] = df["Int8"].astype(np.int64)
    df["Int16"] = df["Int16"].astype(np.int64)
    df["Int32"] = df["Int32"].astype(np.int64)
    assert_frame_equal(expect, df)

    # Numpy UInts
    # FIXME: fastavro reads all uints as np.int64
    columns = ['UInt8', 'UInt16', 'UInt32', 'UInt64']
    expect = pdx.read_avro(tf.name, columns=columns, na_dtypes=False)
    df = dataframe_na_dtypes[columns]
    # Avro does not have the concept of 8 or 16-bit int
    df["UInt8"] = df["UInt8"].astype(np.int64)
    df["UInt16"] = df["UInt16"].astype(np.int64)
    df["UInt32"] = df["UInt32"].astype(np.int64)
    df["UInt64"] = df["UInt64"].astype(np.int64)
    assert_frame_equal(expect, df)

    # Pandas Ints
    columns = ['pdInt8', 'pdInt16', 'pdInt32', 'pdInt64']
    expect = pdx.read_avro(tf.name, columns=columns, na_dtypes=True)
    df = dataframe_na_dtypes[columns]
    # Avro does not have the concept of 8 or 16-bit int
    df["pdInt8"] = df["pdInt8"].astype(pd.Int32Dtype())
    df["pdInt16"] = df["pdInt16"].astype(pd.Int32Dtype())
    assert_frame_equal(expect, df)

    # Pandas UInts
    # fastavro does not seem to support writing 64-bit unsigned ints
    columns = ['pdUInt8', 'pdUInt16', 'pdUInt32']
    expect = pdx.read_avro(tf.name, columns=columns, na_dtypes=True)
    df = dataframe_na_dtypes[columns]
    df["pdUInt8"] = df["pdUInt8"].astype(pd.UInt32Dtype())
    df["pdUInt16"] = df["pdUInt16"].astype(pd.UInt32Dtype())
    assert_frame_equal(expect, df, check_dtype=False)


if __name__ == '__main__':
    pytest.main()
