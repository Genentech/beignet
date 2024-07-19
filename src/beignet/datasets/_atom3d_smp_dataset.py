from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import torch
from pandas import DataFrame
from torch import Tensor

from beignet.transforms import Transform

from ._atom3d_dataset import ATOM3DDataset


class ATOM3DSMPDataset(ATOM3DDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        transform: Union[Callable, Transform, None] = None,
        target_transform: Union[Callable, Transform, None] = None,
    ):
        """
        ATOM3D Small Molecule Properties (SMP) is a dataset of structures
        and energetic, electronic, and thermodynamic properties for 134,000
        stable small organic molecules, obtained from quantum-chemical
        calculations. The task is to predict the molecular properties from
        the ground-state structure.

        Some molecules have been excluded because they failed consistency
        tests or were not properly processed.

        Each sample is a pair of features and a target, where features is
        the molecule’s atomic coordinates and target is a dictionary of the
        following energetic, electronic, and thermodynamic properties:

        .. list-table:: Target
           :widths: 20 20 60
           :header-rows: 1

           * - Key
             - Unit
             - Description
           * - a
             - GHz
             - Rotational constant A
           * - b
             - GHz
             - Rotational constant B
           * - c
             - GHz
             - Rotational constant C
           * - mu
             - Debye
             - Dipole moment
           * - alpha
             - Bohr^3
             - Isotropic polarizability
           * - homo
             - Hartree
             - Energy of Highest occupied molecular orbital (HOMO)
           * - lumo
             - Hartree
             - Energy of Lowest occupied molecular orbital (LUMO)
           * - gap
             - Hartree
             - Gap, difference between LUMO and HOMO
           * - r2
             - Bohr^2
             - Electronic spatial extent
           * - zpve
             - Hartree
             - Zero point vibrational energy
           * - u0
             - Hartree
             - Internal energy at 0 K
           * - u
             - Hartree
             - Internal energy at 298.15 K
           * - h
             - Hartree
             - Enthalpy at 298.15 K
           * - g
             - Hartree
             - Free energy at 298.15 K
           * - cv
             - cal/(mol K)
             - Heat capacity at 298.15 K

        Parameters
        ----------
        root : Union[str, Path]
            The root directory of the dataset.

        download : bool, optional
            If True, download the dataset from the specified source,
            by default `False`.

        transform : Union[Callable, Transform, None], optional
            The transformation function to be applied to the features,
            by default `None`.

        target_transform : Union[Callable, Transform, None], optional
            The transformation function to be applied to the targets,
            by default `None`.
        """
        super().__init__(
            root,
            "raw/QM9/data",
            "https://zenodo.org/record/4911142/files/SMP-raw.tar.gz",
            "SMP",
            checksum="52cc7955c0f80f7dd9faf041e171f405",
            download=download,
        )

        self._transform_fn = transform

        self._target_transform_fn = target_transform

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Tuple[DataFrame], Dict[str, Tensor]]:
        """
        Parameters
        ----------
        index : int
            The index of the item to retrieve from the dataset.

        Returns
        -------
        Tuple[Tuple[DataFrame], Dict[str, Tensor]]
            A tuple containing the features and target of the item.

        """
        item = super().__getitem__(index)

        features = DataFrame(**item["atoms"])

        if self._transform_fn is not None:
            features = self._transform_fn(features)

        target = {}

        for k, v in zip(
            [
                "a",
                "b",
                "c",
                "mu",
                "alpha",
                "homo",
                "lumo",
                "gap",
                "r2",
                "zpve",
                "u0",
                "u",
                "h",
                "g",
                "cv",
            ],
            item["labels"],
            strict=False,
        ):
            target[k] = torch.tensor(v)

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)

        return features, target
