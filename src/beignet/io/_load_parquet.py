from typing import Optional

import pandas as pd
from pyarrow.parquet import ParquetDataset


def load_parquet(
    fpath: str,
    filter_key: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> ParquetDataset:
    """
    Load a parquet file from the specified path.

    Parameters
    ----------
    fpath : str
        The path to the parquet file.
    filter_key : Optional[str], optional
        The partition name to filter on, by default None.
    filter_value : Optional[str], optional
        The value to filter on, by default None.

    Returns
    -------
    pd.DataFrame
        The dataframe loaded from the parquet file.
    """

    filters = None
    if filter_key is not None and filter_value is not None:
        filters = [(filter_key, "in", [filter_value])]

    return ParquetDataset(fpath, filters=filters, use_legacy_dataset=False)


def load_parquet_to_pandas(*args, **kwargs) -> pd.DataFrame:
    """
    Load a parquet file from the specified path and convert it to a pandas dataframe.
    """
    return load_parquet(*args, **kwargs).read().to_pandas(self_destruct=True)
