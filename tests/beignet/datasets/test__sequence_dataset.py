from pathlib import Path

import pytest
from beignet.datasets import SequenceDataset


class TestSequenceDataset:
    @pytest.mark.parametrize(
        "root",
        [
            "./bar/baz",
            "bar/baz",
            Path("./bar/baz"),
            Path("bar/baz"),
        ],
    )
    def test___init__(self, mocker, root):
        resolve = mocker.patch(
            "pathlib.Path.resolve",
            return_value=Path("/foo/bar/baz"),
        )

        dataset = SequenceDataset(root)

        assert isinstance(dataset.root, Path)

        assert dataset.root == Path("/foo/bar/baz")

        resolve.assert_called_once()
