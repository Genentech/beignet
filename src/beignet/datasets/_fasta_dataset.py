import subprocess
from pathlib import Path
from typing import Callable, Tuple, TypeVar

import numpy

from beignet.io import ThreadSafeFile

from ._sized_sequence_dataset import SizedSequenceDataset

T = TypeVar("T")


class FASTADataset(SizedSequenceDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        index: bool = True,
        transform: Callable[[T], T] | None = None,
    ) -> None:
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError

        self._thread_safe_file = ThreadSafeFile(root, open)

        self._index = Path(f"{self.root}.index.npy")

        if index:
            if self._index.exists():
                self.offsets, sizes = numpy.load(str(self._index))
            else:
                self.offsets, sizes = self._build_index()

                numpy.save(str(self._index), numpy.stack([self.offsets, sizes]))
        else:
            self.offsets, sizes = self._build_index()

        self._transform_fn = transform

        super().__init__(self.root, sizes)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        x = self.get(index)

        if self._transform_fn:
            x = self._transform_fn(x)

        return x

    def __len__(self) -> int:
        return self.offsets.size

    def get(self, index: int) -> str:
        self._thread_safe_file.seek(self.offsets[index])

        if index == len(self) - 1:
            data = self._thread_safe_file.read()
        else:
            data = self._thread_safe_file.read(
                self.offsets[index + 1] - self.offsets[index],
            )

        description, *sequence = data.split("\n")

        return "".join(sequence)

    def _build_index(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        # TODO: rewrite in Rust (using `libripgrep`) or similar to remove
        #  dependency on `grep` and `awk`. — Allen (Tuesday, November 29, 2022)
        return (
            numpy.fromstring(
                subprocess.check_output(
                    f"cat {self.root} "
                    f"| tqdm --bytes --total $(wc -c < {self.root})"
                    "| grep --byte-offset '^>' -o | cut -d: -f1",
                    shell=True,
                ),
                dtype=numpy.int64,
                sep=" ",
            ),
            numpy.fromstring(
                subprocess.check_output(
                    f"cat {self.root} "
                    f"| tqdm --bytes --total $(wc -c < {self.root})"
                    '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' '
                    "| tail -n+2 | awk "
                    "'{print length($1)}'",
                    shell=True,
                ),
                dtype=numpy.int64,
                sep=" ",
            ),
        )
