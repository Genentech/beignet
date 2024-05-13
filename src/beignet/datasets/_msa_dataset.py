from pathlib import Path
from typing import Callable

import numpy
import pooch
import torch
from torch import Tensor
from torch.utils.data import Dataset


def make_similarity_matrices(*args):
    return


class MSADataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        if isinstance(root, str):
            root = Path(root)

        name = self.__class__.__name__

        if download:
            pooch.retrieve(
                "https://files.ipd.uw.edu/krypton/data_unalign.npz",
                fname=f"{name}.npz",
                known_hash="9cc22e381619b66fc353c079221fd02450705d4e3ee23e4e23a052b6e70a95ec",
                path=root / name,
            )

        self.all_data = numpy.load(root / name / f"{name}.npz", allow_pickle=True)

        all_sequences = []
        all_alignments = []
        all_sizes = []
        all_matrices = []

        # process each subset
        for subset in self.all_data.files:
            data = self.all_data[subset].tolist()

            # pad sequences
            sequences = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(data["ms"]),
                0.0,
            )
            sequences = torch.concatenate(
                [
                    torch.eye(torch.max(sequences) + 1),
                    torch.zeros([1, torch.max(sequences) + 1]),
                ],
            )[sequences]

            reference_sequence, sequences = sequences[0].unsqueeze(0), sequences[1:]
            all_sequences.append(sequences)

            sizes = torch.tensor([len(seq) for seq in sequences])
            all_sizes.append(sizes)

            # pad alignments
            alignments = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(data["aln"]),
                0.0,
            )

            alignments = torch.concatenate(
                [
                    torch.eye(torch.max(alignments) + 1),
                    torch.zeros([1, torch.max(alignments) + 1]),
                ],
            )[alignments]

            _, alignments = alignments[0], alignments[1:]  # ignore first alignment
            all_alignments.append(alignments)

            matrices = make_similarity_matrices(
                sequences, reference_sequence
            )  # TODO (Edith): make matrices
            all_matrices.append(matrices)

        self.sequences = torch.stack(all_sequences, dim=1)
        self.alignments = torch.stack(all_alignments, dim=1)
        self.sizes = torch.stack(all_sizes, dim=1)
        self.matrices = torch.stack(all_matrices, dim=1)

        self.transform = transform

        self.target_transform = target_transform

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        inputs = self.matrices[index], self.sizes[index]

        if self.transform:
            inputs = self.transform(*inputs)

        target = self.alignments[index]

        if self.target_transform:
            target = self.target_transform(target)

        return inputs, target
