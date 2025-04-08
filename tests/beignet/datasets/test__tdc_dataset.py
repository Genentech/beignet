from pathlib import Path
from unittest.mock import patch

from pandas import DataFrame

from beignet.datasets._tdc_dataset import TDCDataset


class TestTDCDataset:
    @patch("pooch.retrieve")
    @patch("pandas.read_csv")
    def test___init__(self, mock_read_csv, mock_retrieve):
        mock_read_csv.return_value = DataFrame(
            {
                "x": [None, "A", "B"],
                "y": [0, None, 2],
            }
        )

        dataset = TDCDataset(
            root="./foo/bar",
            download=True,
            identifier=123,
            suffix="csv",
            checksum="md5:checksum",
            x_keys=["x"],
            y_keys=["y"],
        )

        mock_retrieve.assert_called_once_with(
            "https://dataverse.harvard.edu/api/access/datafile/123",
            fname="TDCDataset.csv",
            known_hash="md5:checksum",
            path=Path("./foo/bar") / "TDCDataset",
            progressbar=True,
        )
        mock_read_csv.assert_called_once_with(
            Path("./foo/bar") / "TDCDataset" / "TDCDataset.csv", sep=None
        )
        assert dataset._x_keys == ["x"]
        assert dataset._y_keys == ["y"]

        assert list(dataset._x) == [("B",)]
        assert list(dataset._y) == [(2.0,)]
