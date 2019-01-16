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
