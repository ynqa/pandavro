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

AVRO can represent the following kinds of types:
- Primitive types (`null`, `bool`, `int` etc.)
- Complex types (records, arrays, maps etc.)
- Logical types (annotated primitive/complex type to represent e.g. datetime)

When converting to AVRO, pandavro will try to infer the schema. It will output a non-nested schema *without any indexes* set on the dataframe and it will also not try to infer if any column can be nullable so *all columns are set as nullable*, i.e. a boolean will be encoded in AVRO schema as `['null', 'bool']`.

Pandavro can handle these primitive types:

| Numpy type | AVRO primitive type |
|------------|-----------|
| np.bool_   | boolean   |
| np.int8 or np.int16 or np.int32 | int |
| np.int64 | long |
| np.float32 | float |
| np.float64 | double |
| np.object | string |

And these logical types:

| Numpy type | AVRO logical type |
|------------|-------------------|
| np.datetime64 or pd.core.dtypes.dtypes.DatetimeTZDtype | timestamp-micros |

Note that the timestamp must not contain any timezone, i.e. it must be naive.

If you don't want pandavro to infer this schema but instead define it yourself, pass it using the `schema` kwarg to `to_avro`.


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
