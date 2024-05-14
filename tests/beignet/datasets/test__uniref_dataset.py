from unittest import mock

from beignet.datasets import UniRef50Dataset


class TestUnirefDataset:
    @mock.patch("beignet.io.download_and_extract_archive", return_value=True)
    def test_init(self, download_func):
        dataset = UniRef50Dataset(root="/tmp/data", download=True)

        assert download_func.called_once()
        assert isinstance(dataset, UniRef50Dataset)
