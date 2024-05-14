import subprocess
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy

from beignet.io import ThreadSafeFile

from ._sized_sequence_dataset import SizedSequenceDataset


class FASTADataset(SizedSequenceDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        cache_sequence_indicies: bool = True,
        transform_fn: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError

        self._thread_safe_file = ThreadSafeFile(root, open)

        self.cache = Path(f"{self.root}.index.npy")

        if cache_sequence_indicies:
            if self.cache.exists():
                self.offsets, sizes = numpy.load(str(self.cache))
            else:
                self.offsets, sizes = self._build_index()

                numpy.save(str(self.cache), numpy.stack([self.offsets, sizes]))
        else:
            self.offsets, sizes = self._build_index()

        self._transform_fn = transform_fn

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
                    f"cat {self.root} | tqdm --bytes --total $(wc -c < {self.root})"
                    "| grep --byte-offset '^>' -o | cut -d: -f1",
                    shell=True,
                ),
                dtype=numpy.int64,
                sep=" ",
            ),
            numpy.fromstring(
                subprocess.check_output(
                    f"cat {self.root} | tqdm --bytes --total $(wc -c < {self.root})"
                    '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' | tail -n+2 | awk '
                    "'{print length($1)}'",
                    shell=True,
                ),
                dtype=numpy.int64,
                sep=" ",
            ),
        )
