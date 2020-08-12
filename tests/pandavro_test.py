import pytest
import numpy as np
import pandas as pd
import pandavro as pdx
from tempfile import NamedTemporaryFile
from pandas.util.testing import assert_frame_equal
from io import BytesIO


@pytest.fixture
def dataframe():
    return pd.DataFrame({"Boolean": [True, False, True, False, True, False, True, False],
                         "DateTime64": pd.date_range('20190101', '20190108', freq="1D", tz="UTC"),
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
    assert expect == pdx.__schema_infer(dataframe, times_as_micros=True)


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
    assert expect == pdx.__schema_infer(dataframe, times_as_micros=False)


def test_fields_infer(dataframe):
    expect = [
        {'type': ['null', 'boolean'], 'name': 'Boolean'},
        {'type': ['null', {'logicalType': 'timestamp-micros', 'type': 'long'}],
            'name': 'DateTime64'},
        {'type': ['null', 'double'], 'name': 'Float64'},
        {'type': ['null', 'long'], 'name': 'Int64'},
        {'type': ['null', 'string'], 'name': 'String'},
    ]
    assert expect == pdx.__fields_infer(dataframe)


def test_buffer_e2e(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    with open(tf.name, 'rb') as f:
        expect = pdx.read_avro(BytesIO(f.read()))
        # expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    assert_frame_equal(expect, dataframe)


def test_file_path_e2e(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    expect = pdx.read_avro(tf.name)
    # expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    assert_frame_equal(expect, dataframe)


def test_delegation(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe)
    expect = pdx.from_avro(tf.name)
    # expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    assert_frame_equal(expect, dataframe)


def test_append(dataframe):
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe[0:int(dataframe.shape[0] / 2)])
    pdx.to_avro(tf.name, dataframe[int(dataframe.shape[0] / 2):], append=True)
    expect = pdx.from_avro(tf.name)
    # expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
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
    # expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    df = dataframe.drop(columns, axis=1)
    assert_frame_equal(expect, df)
    # specify index
    index = 'String'
    expect = pdx.read_avro(tf.name, index=index)
    # expect['DateTime64'] = expect['DateTime64'].astype(np.dtype('datetime64[ns]'))
    df = dataframe.set_index(index)
    assert_frame_equal(expect, df)


@pytest.fixture
def dataframe_na_dtypes():
    return pd.DataFrame({
        "Boolean": [True, False, True, False, True, False, True, False],
        "pdBoolean": pd.Series([True, False, True, False, True, False, True, False]).astype(pd.BooleanDtype()),
        "DateTime64": pd.date_range('20190101', '20190108', freq="1D", tz="UTC"),
        "Float64": np.random.randn(8),
        "Int64": np.random.randint(0, 10, 8),
        "pdInt64": pd.Series(list(np.random.randint(0, 10, 7)) + [None]).astype(pd.Int64Dtype()),
        "String": ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
        "pdString": pd.Series(['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar']).astype(pd.StringDtype())
    })


def test_advanced_dtypes(dataframe_na_dtypes):
    "Should be able to write and read Pandas 1.0 NaN-compatible dtypes"
    tf = NamedTemporaryFile()
    pdx.to_avro(tf.name, dataframe_na_dtypes)

    # Bools and datetime
    columns = ['Boolean', 'pdBoolean', 'DateTime64']
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


def test_benchmark_advanced_dtypes(dataframe):
    "Should not be much slower for basic dtype dataframes with Pandas NA-dtypes preprocessing"
    import timeit
    reps = 1000

    t1 = timeit.timeit(
        "pdx.to_avro(filename, df)",
        globals=dict(pdx=pdx, filename=NamedTemporaryFile().name, df=dataframe),
        number=reps
    )

    t2 = timeit.timeit(
        "pdx.to_avro(filename, df, _test_preprocess_off=True)",
        globals=dict(pdx=pdx, filename=NamedTemporaryFile().name, df=dataframe),
        number=reps
    )

    # 20% was arbitrarily chosen to give some leeway in this slightly random benchmark
    # Observed differences are very small
    assert abs(t1 - t2) / min(t1, t2) < .2, "Performance difference is not below 20%, " \
                                            "{:.3f}s with and {:.3f}s without".format(t1, t2)


if __name__ == '__main__':
    pytest.main()
