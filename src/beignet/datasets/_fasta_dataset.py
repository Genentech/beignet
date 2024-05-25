import subprocess
from os import PathLike
from pathlib import Path
from typing import Callable, TypeVar

import numpy

from beignet.io import ThreadSafeFile

from ..transforms import Transform
from ._sized_sequence_dataset import SizedSequenceDataset

T = TypeVar("T")


class FASTADataset(SizedSequenceDataset):
    def __init__(
        self,
        root: str | PathLike,
        *,
        transform: Callable | Transform | None = None,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()

        if not self.root.exists():
            raise FileNotFoundError

        self.data = ThreadSafeFile(self.root, open)

        offsets = Path(f"{self.root}.offsets.npy")

        if offsets.exists():
            self.offsets, sizes = numpy.load(f"{offsets}")
        else:
            self.offsets = numpy.fromstring(
                subprocess.check_output(
                    f"cat {self.root} "
                    f"| tqdm --bytes --total $(wc -c < {self.root}) "
                    f"| grep --byte-offset '^>' -o "
                    f"| cut -d: -f1",
                    shell=True,
                ),
                dtype=numpy.int64,
                sep=" ",
            )

            sizes = numpy.fromstring(
                subprocess.check_output(
                    f"cat {self.root} "
                    f"| tqdm --bytes --total $(wc -c < {self.root})"
                    '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' '
                    "| tail -n+2 "
                    "| awk '{print length($1)}'",
                    shell=True,
                ),
                dtype=numpy.int64,
                sep=" ",
            )

            numpy.save(f"{offsets}", numpy.stack([self.offsets, sizes]))

        self.transform = transform

        super().__init__(self.root, sizes)

    def __getitem__(self, index: int) -> (str, str):
        x = self.get(index)

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return self.offsets.size

    def get(self, index: int) -> (str, str):
        self.data.seek(self.offsets[index])

        if index == len(self) - 1:
            data = self.data.read()
        else:
            data = self.data.read(self.offsets[index + 1] - self.offsets[index])

        description, *sequence = data.split("\n")

        return "".join(sequence), description
