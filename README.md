# pandavro

[![Build Status](https://travis-ci.org/ynqa/pandavro.svg?branch=master)](https://travis-ci.org/ynqa/pandavro)

The interface between Apache Avro and pandas DataFrame.

## Installation

`pandavro` is available to install from [PyPI](https://pypi.org/project/pandavro/).

```bash
$ pip install pandavro
```

## Description

It prepares like pandas APIs:

- `read_avro`
    - Read the records from Avro file and fit them into pandas DataFrame using [fastavro](https://github.com/tebeka/fastavro).
- `to_avro`
    - Write the rows of pandas DataFrame to Avro file with the original schema infer.
    
## What can and can't pandavro do?

Avro can represent the following kinds of types:
- Primitive types (`null`, `bool`, `int` etc.)
- Complex types (records, arrays, maps etc.)
- Logical types (annotated primitive/complex type to represent e.g. datetime)

When converting to Avro, pandavro will try to infer the schema. It will output a non-nested schema *without any indexes* set on the dataframe and it will also not try to infer if any column can be nullable so *all columns are set as nullable*, i.e. a boolean will be encoded in Avro schema as `['null', 'bool']`.

Pandavro can handle these primitive types:

| Numpy/pandas type                             | Avro primitive type |
|-----------------------------------------------|---------------------|
| np.bool_                                      | boolean             |
| np.float32                                    | float               |
| np.float64                                    | double              |
| np.unicode_                                   | string              |
| np.object_                                    | string              |
| np.int8, np.int16, np.int32                   | int                 |
| np.uint8, np.uint16, np.uint32                | int                 |
| np.int64, np.uint64                           | long                |
| pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype*   | int                 |
| pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype*| "unsigned" int      |
| pd.Int64Dtype*                                | long                |
| pd.UInt64Dtype*                               | "unsigned" long     |
/ pd.StringDtype**                              / string              /
/ pd.BooleanDtype**                             / boolean             /

\* Pandas 0.24 added support for nullable integers, which we can easily represent in Avro. We represent the unsigned versions of these integers by adding the non-standard "unsigned" flag as such: `{'type': 'int', 'unsigned': True}`.

\** Pandas 1.0.0 added support for nullable string and boolean datatypes.

And these logical types:

| Numpy/pandas type                               | Avro logical type |
|-------------------------------------------------|-------------------|
| np.datetime64, pd.DatetimeTZDtype, pd.Timestamp | timestamp-micros  |

Note that the timestamp must not contain any timezone (it must be naive) because Avro does not support timezones.

If you don't want pandavro to infer this schema but instead define it yourself, pass it using the `schema` kwarg to `to_avro`.

## Loading Pandas nullable datatypes
The nullable datatypes indicated in the table above are easily written to Avro, but loading them introduces ambiguity as we can use either the old, default or these new datatypes. We solve this by using a special keyword when loading to force conversion to these new NA-supporting datatypes (`support_na=True`).

This is *different* from [convert_dtypes](https://pandas.pydata.org/docs/whatsnew/v1.0.0.html#convert-dtypes-method-to-ease-use-of-supported-extension-dtypes) as it does not infer the datatype based on the actual values, but it looks at the Avro schema so is deterministic and not dependent on the actual values.

## Example

```python
import os
import numpy as np
import pandas as pd
import pandavro as pdx

OUTPUT_PATH='{}/example.avro'.format(os.path.dirname(__file__))


def main():
    df = pd.DataFrame({"Boolean": [True, False, True, False],
                       "Float64": np.random.randn(4),
                       "Int64": np.random.randint(0, 10, 4),
                       "String": ['foo', 'bar', 'foo', 'bar'],
                       "DateTime64": [pd.Timestamp('20190101'), pd.Timestamp('20190102'),
                                      pd.Timestamp('20190103'), pd.Timestamp('20190104')]})

    pdx.to_avro(OUTPUT_PATH, df)
    saved = pdx.read_avro(OUTPUT_PATH)
    print(saved)


if __name__ == '__main__':
    main()
```
