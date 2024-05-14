import pandas as pd
import pytest
from beignet.io._load_parquet import load_parquet_to_pandas


@pytest.fixture
def df():
    return pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})


@pytest.mark.parametrize(
    "filter_key, filter_value",
    [
        (None, None),
    ],
)
def test_load_parquet(filter_key, filter_value, df, tmpdir):
    fpath = tmpdir.join("test.parquet")
    df.to_parquet(str(fpath))

    loaded_df = load_parquet_to_pandas(str(fpath), filter_key, filter_value)
    assert loaded_df.equals(df)
