import sys
import unittest.mock
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from beignet.datasets import UniProtDataset


class TestUniProtDataset:
    @pytest.mark.skipif(sys.platform == "win32", reason="windows")
    def test___init__(self):
        with NamedTemporaryFile() as descriptor:
            with unittest.mock.patch(
                "pooch.retrieve",
                lambda a, b, c, d, **_: descriptor.name,
            ):
                dataset = UniProtDataset(
                    "https://example.com",
                    descriptor.name,
                )

                assert dataset.root == Path(descriptor.name).resolve()

                assert dataset.transform is None

                assert dataset.target_transform is None
