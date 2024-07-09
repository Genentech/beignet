from beignet.datasets import SizedSequenceDataset


class TestSizedSequenceDataset:
    def test___init__(self, mocker):
        mock_sequence_dataset_init = mocker.patch(
            "beignet.datasets.SequenceDataset.__init__",
            return_value=None,
        )

        sizes = [1, 2, 3]

        dataset = SizedSequenceDataset("/foo/bar", sizes)

        mock_sequence_dataset_init.assert_called_once_with("/foo/bar")

        assert dataset.sizes == sizes

    def test___len__(self):
        sizes = [1, 2, 3]

        assert len(SizedSequenceDataset("/foo/bar", sizes)) == len(sizes)
