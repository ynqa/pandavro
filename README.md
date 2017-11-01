# pandavro

[![Build Status](https://travis-ci.org/ynqa/pandavro.svg?branch=master)](https://travis-ci.org/ynqa/pandavro)

The interface between Apache Avro and pandas DataFrame.

## Installation

`pandavro` is available to install from [PyPI](https://pypi.python.org/pypi).

```bash
$ pip install pandavro
```

## Description

It prepares like pandas APIs:

- `from_avro`
    - Read the records from Avro file and fit them into pandas DataFrame using [fastavro](https://github.com/tebeka/fastavro).
- `to_avro`
    - Write the rows of pandas DataFrame to Avro file with the original schema infer.

## Example

```python
import pandavro as pdx


def main():
    weather = pdx.from_avro('weather.avro')

    print(weather)

    pdx.to_avro('weather_out.avro', weather)

if __name__ == '__main__':
    main()

```