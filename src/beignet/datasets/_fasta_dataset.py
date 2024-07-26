from os import PathLike
from pathlib import Path
from typing import Callable, Tuple, TypeVar

import numpy
import tqdm

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
            self.root = Path(root)

        self.root = self.root.resolve()

        if not self.root.exists():
            raise FileNotFoundError

        self.data = ThreadSafeFile(self.root, open)

        offsets = Path(f"{self.root}.offsets.npy")

        if offsets.exists():
            self.offsets, sizes = numpy.load(f"{offsets}")
        else:
            self.offsets, sizes = self._build_index()

            numpy.save(f"{offsets}", numpy.stack([self.offsets, sizes]))

        self.transform = transform

        super().__init__(self.root, sizes)

    def __getitem__(self, index: int) -> Tuple[str, str]:
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

    def _build_index(self) -> (numpy.ndarray, numpy.ndarray):
        with open(self.root, "r") as file:
            content = file.read()

        offsets, sizes = [], []

        current_offset, current_size = 0, 0

        parsing = False

        for sequence in tqdm.tqdm(content.splitlines(keepends=True)):
            characters = len(sequence)

            if sequence.startswith(">"):
                if parsing:
                    sizes = [*sizes, current_size]

                    current_size = 0

                offsets = [*offsets, current_offset]

                parsing = True
            elif parsing:
                current_size = current_size + len(sequence.rstrip("\n"))

            current_offset = current_offset + characters

        if parsing:
            sizes = [*sizes, current_size]

        offsets = numpy.array(offsets, dtype=numpy.int64)

        sizes = numpy.array(sizes, dtype=numpy.int64)

        return offsets, sizes
