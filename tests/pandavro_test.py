import pytest
import numpy as np
import pandas as pd
import pandavro as pdx
from tempfile import NamedTemporaryFile
from pandas.testing import assert_frame_equal
from io import BytesIO


@pytest.fixture
def dataframe():
    return pd.DataFrame({"Boolean": [True, False, True, False, True, False, True, False],
                         "DateTime64": [pd.Timestamp('20190101'), pd.Timestamp('20190102'),
                                        pd.Timestamp('20190103'), pd.Timestamp('20190104'),
                                        pd.Timestamp('20190105'), pd.Timestamp('20190106'),
                                        pd.Timestamp('20190107'), pd.Timestamp('20190108')],
                         "Float64": np.random.randn(8),
                         "Int64": np.random.randint(0, 10, 8),
                         "String": ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar']})


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
    ]
    assert expect == pdx.__fields_infer(dataframe, nested_record_names={})


def test_buffer_e2e(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    with open(tf.name, 'rb') as f:
        expect = pdx.read_avro(BytesIO(f.read()))
        expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    assert_frame_equal(expect, dataframe)


def test_file_path_e2e(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    expect = pdx.read_avro(tf.name)
    expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    assert_frame_equal(expect, dataframe)


def test_delegation(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    expect = pdx.from_avro(tf.name)
    expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    assert_frame_equal(expect, dataframe)


def test_append(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe[0:int(dataframe.shape[0] / 2)])
    pdx.to_avro(tf.name, dataframe[int(dataframe.shape[0] / 2):], append=True)
    expect = pdx.from_avro(tf.name)
    expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    assert_frame_equal(expect, dataframe)


def test_dataframe_kwargs(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    # include columns
    columns = ['Boolean', 'Int64']
    expect = pdx.read_avro(tf.name, columns=columns)
    df = dataframe[columns]
    assert_frame_equal(expect, df)
    # exclude columns
    columns = ['String', 'Boolean']
    expect = pdx.read_avro(tf.name, exclude=columns)
    expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    df = dataframe.drop(columns, axis=1)
    assert_frame_equal(expect, df)
    # specify index
    index = 'String'
    expect = pdx.read_avro(tf.name, index=index)
    expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    df = dataframe.set_index(index)
    assert_frame_equal(expect, df)


if __name__ == '__main__':
    pytest.main()
