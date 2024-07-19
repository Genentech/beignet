import os.path
from pathlib import Path
from typing import Callable, Union

import torch
from Bio.PDB import PDBParser
from torch import Tensor

import beignet.io
from beignet.transforms import Transform

from ._parquet_dataset import ParquetDataset


class SKEMPIDataset(ParquetDataset):
    """
    The Structural Kinetic and Energetic database of Mutant Protein
    Interactions (SKEMPI) database is a compilation of experimental data on
    the thermodynamics of mutations in protein-protein interactions. The
    database includes protein names, protein structures from the Protein
    Data Bank (PDB), mutation information, and the change in free energy
    upon mutation. The change in free energy gives an indication of how the
    mutation affects the binding affinity of the two proteins.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        sequence_transform_fn: Union[Callable, Transform, None] = None,
        structure_transform_fn: Union[Callable, Transform, None] = None,
        target_transform: Union[Callable, Transform, None] = None,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param download: If ``True``, download the dataset to the :attr:`root`
            directory (default: ``False``). If the dataset is already
            downloaded, it is not redownloaded.

        :param sequence_transform_fn: A ``Callable`` or ``Transform`` that maps
            sequences to transformed sequences (default: ``None``).

        :param structure_transform_fn: A ``Callable`` or ``Transform`` that
            maps structures to transformed structures (default: ``None``).

        :param target_transform: ``Callable`` or ``Transform`` that maps a
            target to a transformed target (default: ``None``).
        """
        if isinstance(root, str):
            root = Path(root).resolve()

        self._root = root

        if download:
            beignet.io.download(
                source="s3://beignet-data-dev/designdb/lake/thirdparty/skempi/cc5952a4a37f4f1fbe14ce484a00eb87_0.snappy.parquet",
                destination=self._root / "SKEMPI-v2.0",
                filename="SKEMPI-v2.0.parquet",
            )

            beignet.io.download_and_extract_archive(
                resource="https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz",
                source=self._root,
                destination=self._root,
                name="SKEMPI-v2.0.tar.gz",
                remove_archive=True,
            )

        super().__init__(
            self._root / "SKEMPI-v2.0",
            self._root / "SKEMPI-v2.0" / "SKEMPI-v2.0.parquet",
        )

        self._sequence_transform_fn = sequence_transform_fn

        self._structure_transform_fn = structure_transform_fn

        self._target_transform_fn = target_transform

        self._data = self._data.dropna(
            subset=[
                "affinity_antigen_sequence",
                "affinity_pkd",
                "fv_heavy",
                "fv_light",
            ],
        )

        self._parser = PDBParser()

        self._structure_paths = [*self._root.glob("PDBs/*.pdb")]

    def __getitem__(
        self,
        index: int,
    ) -> (((str, str, str), (Tensor, [str])), (float, ...)):
        """
        :param index: index of the record to return.

        :returns: A pair of the form:

            .. math::

                \\left(\\text{antibodies},\\;\\text{targets}\\right).

            Each antibody in :math:`\\text{antibodies}` is a pair of the form:

            .. math::

                \\left(\\text{sequences},\\;\\text{structures}\\right).

            :math:`\\text{sequences} `is a :math:`3`-tuple of the form:

            .. math::

                \\left(\\text{VH},\\;\\text{VL},\\;\\text{Ag}\\right)

            where `\\text{VH}` is a ``str`` that represents the
            immunoglobulin heavy chain variable region sequence, `\\text{
            VL}` is a ``str`` that represents the immunoglobulin light chain
            variable region sequence, and $\\text{Ag}` is a ``str`` that
            represents the antigen sequence.

            An antibody is made up of two heavy chains and two light chains.
            Each heavy and light chain has a variable (:math:`V`) region and
            a constant (:math:`C`) region. The variable regions of the heavy
            and light chains form the antigen-binding site of the antibody.
            Each variable region is unique and gives the antibody its
            specificity for binding to a particular antigen. The heavy and
            light chain variable regions are named for their extensive
            sequence variability among different antibodies. This
            variability allows the immune system to produce antibodies that
            can specifically recognize and bind to a vast array of antigens.

            Antigens are molecules capable of stimulating an immune
            response. They are typically proteins or polysaccharides. This
            includes c omponents of bacterial cell walls, capsules, pili,
            and bacterial flagella, as well as proteins in viruses.

            The immune system recognizes antigens as foreign and mounts an
            immune response against them. Antigens are recognized by
            specific antibodies, which bind to the antigen. This binding can
            neutralize the antigen, mark it for destruction by other immune
            cells, or trigger other types of immune responses. Each type of
            antibody recognizes and binds to a specific antigen; this
            specificity is determined by the variable regions of the
            antibody's heavy and light chains.

            :math:`\\text{pKd}` is the negative logarithm of the
            dissociation constant (:math:`\\text{Kd}`). The dissociation
            constant is a measure of how tightly a ligand (e.g., a drug)
            binds to a receptor. The smaller the ``Kd`` value, the tighter
            or stronger the binding between the ligand and its receptor.
            Because :math:`\\text{pKd}` is the negative logarithm of
            :math:`\\text{Kd}`, a larger :math:`\\text{pKd}` value therefore
            represents stronger binding affinity. The :math:`\\text{pKd}`
            value is commonly used in pharmacology and medicinal chemistry
            because it allows easier comparison of binding affinities across
            different ligand-receptor pairs. It’s an important metric when
            assessing the potential efficacy of a drug.
        """
        item = super().__getitem__(index)

        sequence = (
            item["fv_heavy"],
            item["fv_light"],
            item["affinity_antigen_sequence"],
        )

        if self._sequence_transform_fn is not None:
            sequence = self._sequence_transform_fn(sequence)

        name, _ = os.path.splitext(
            os.path.basename(
                self._structure_paths[index],
            ),
        )

        structure = self._parser.get_structure(
            name,
            self._structure_paths[index],
        )

        atomic_coordinates = []

        residue_names = []

        atom_names = []

        alternate_location_indicators = []

        for atom in [*structure.get_atoms()]:
            atomic_coordinates = [
                *atomic_coordinates,
                torch.from_numpy(atom.coord),
            ]

            (
                _,
                _,
                residue_name,
                atom_name,
                alternate_location_indicator,
            ) = atom.get_full_id()

            residue_names = [
                *residue_names,
                residue_name,
            ]

            atom_names = [*atom_names, atom_name]

            alternate_location_indicator, _ = alternate_location_indicator

            alternate_location_indicators = [
                *alternate_location_indicators,
                alternate_location_indicator,
            ]

        structure = (
            torch.stack(atomic_coordinates),
            residue_names,
        )

        if self._structure_transform_fn is not None:
            structure = self._structure_transform_fn(sequence)

        target = item["affinity_pkd"]

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(sequence)

        return (sequence, structure), target
