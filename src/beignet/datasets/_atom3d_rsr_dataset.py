from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import torch
from pandas import DataFrame
from torch import Tensor

from beignet.transforms import Transform

from ._atom3d_dataset import ATOM3DDataset


class ATOM3DRSRDataset(ATOM3DDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        transform: Union[Callable, Transform, None] = None,
        target_transform: Union[Callable, Transform, None] = None,
    ):
        """
        The ATOM3D RNA Structure Ranking (RSR) task predicts the
        three-dimensional structure of an RNA molecule, given its sequence.
        A total of 21 RNAs are included, which consist of the first 21 RNAs
        from the RNA-Puzzles competition (Cruz et al., 2011).

        This problem is prhased as candidate ranking. For each RNA,
        candidate structural models are generated using FARFAR2 (“Silly Boy”
        Watkins et al., 2020) and calculate each candidate’s atoms’ root
        mean squared deviation (RMSD) to the experimentally determined
        structure.

        Each sample is a pair of features and a target, where features is
        the molecule’s atomic coordinates and target is a dictionary of the
        following scores:

        .. list-table:: Target
           :widths: 20 80
           :header-rows: 1

           * - Key
             - Description
           * - score
             -
           * - fa_atr
             -
           * - fa_rep
             -
           * - fa_intra_rep
             -
           * - lk_nonpolar
             -
           * - fa_elec_rna_phos_phos
             -
           * - rna_torsion
             -
           * - suiteness_bonus
             -
           * - rna_sugar_close
             -
           * - fa_stack
             -
           * - stack_elec
             -
           * - geom_sol_fast
             -
           * - hbond_sr_bb_sc
             -
           * - hbond_lr_bb_sc
             -
           * - hbond_sc
             -
           * - ref
             -
           * - free_suite
             -
           * - free_2HOprime
             -
           * - intermol
             -
           * - other_pose
             -
           * - loop_close
             -
           * - linear_chainbreak
             -
           * - rms
             -
           * - rms_stem
             -
           * - time
             -
           * - N_WC
             -
           * - N_NWC
             -
           * - N_BS
             -
           * - N_BP
             -
           * - natWC
             -
           * - natNWC
             -
           * - natBP
             -
           * - f_natWC
             -
           * - f_natNWC
             -
           * - f_natBP
             -

        Parameters
        ----------
        root : Union[str, Path]
            The root directory of the dataset.

        download : bool, optional
            If `True`, download the dataset from the specified source,
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
            "raw/candidates/data",
            "https://zenodo.org/record/4961085/files/RSR-raw.tar.gz",
            "RSR",
            checksum="68830ab0ab95cf3d218785a4e4e7669c",
            download=download,
        )

        self._transform_fn = transform

        self._target_transform_fn = target_transform

    def __getitem__(self, index: int) -> Tuple[DataFrame, Dict[str, Tensor]]:
        """
        Parameters
        ----------
        index : int
            The index of the item to retrieve from the dataset.

        Returns
        -------
        Tuple[DataFrame, Dict[str, Tensor]]
            A tuple containing the features and target of the item.
        """
        item = super().__getitem__(index)

        features = DataFrame(**item["atoms"])

        if self._transform_fn is not None:
            features = self._transform_fn(features)

        target = item["scores"]

        for k, v in target.items():
            target[k] = torch.tensor(v)

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)

        return features, target
