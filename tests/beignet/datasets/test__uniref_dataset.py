from unittest import mock

from beignet.datasets import UniRef50Dataset


class TestUnirefDataset:
    @mock.patch("beignet.io._download.download_and_extract_archive", return_value=True)
    @mock.patch("beignet.datasets.UniRef50Dataset.__init__", return_value=None)
    def test_init(self, download_func, uniref_init):
        dataset = UniRef50Dataset(root="/tmp/data", download=True)

        assert download_func.called_once()
        assert isinstance(dataset, UniRef50Dataset)
