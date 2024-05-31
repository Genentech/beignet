from pathlib import Path
from typing import Callable

import numpy
import pooch
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ._smurf_dataset_constants import (
    FAMILIES_TEST,
    FAMILIES_TRAIN,
    NUM_SEQUENCES_TEST,
    NUM_SEQUENCES_TRAIN,
)


class SMURFDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
        train: bool = True,
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

        self.all_data = numpy.load(
            root / name / f"{name}.npz", allow_pickle=True, mmap_mode="r"
        )

        if train:
            families = FAMILIES_TRAIN
            num_sequences = NUM_SEQUENCES_TRAIN
        else:
            families = FAMILIES_TEST
            num_sequences = NUM_SEQUENCES_TEST

        self.all_sequences = torch.zeros([num_sequences, 583])
        self.all_references = torch.zeros([num_sequences, 583])
        self.all_alignments = torch.zeros([num_sequences, 583])
        self.all_sizes = torch.zeros([num_sequences, 1])

        idx = 0

        for family in families:
            data = self.all_data[family].tolist()

            # sequences
            sequences = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(data["ms"]), 0.0
            )
            reference_sequence, sequences = sequences[0], sequences[1:]

            chunk = torch.zeros([sequences.shape[0], 583])
            chunk[:, : sequences.shape[1]] = sequences
            self.all_sequences[idx : idx + sequences.shape[0], :] = chunk

            chunk = torch.zeros([sequences.shape[0], 583])
            chunk[:, : sequences.shape[1]] = reference_sequence.repeat(
                (sequences.shape[0], 1)
            )
            self.all_references[idx : idx + sequences.shape[0], :] = chunk

            # alignments
            alignments = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(data["aln"]), 0.0
            )
            _, alignments = alignments[0], alignments[1:]  # discard the first alignment

            chunk = torch.zeros([alignments.shape[0], 583])
            chunk[:, : sequences.shape[1]] = alignments
            self.all_alignments[idx : idx + sequences.shape[0], :] = chunk

            # sizes
            self.all_sizes[idx : idx + sequences.shape[0], :] = torch.tensor(
                [len(seq) for seq in sequences]
            ).unsqueeze(1)

            idx += sequences.shape[0]

        self.transform = transform

        self.target_transform = target_transform

    def __len__(self):
        return self.all_sequences.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        inputs = (
            self.all_sequences[index],
            self.all_references[index],
            self.all_sizes[index],
        )

        if self.transform:
            inputs = self.transform(*inputs)

        target = self.all_alignments[index]

        if self.target_transform:
            target = self.target_transform(target)

        return inputs, target
