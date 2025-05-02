from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from .. import ResidueArray, rmsd, superimpose
from ..residue_selectors import CDRResidueSelector, ChainSelector


@dataclass
class AntibodyRMSDDescriptors:
    fv_heavy_rmsd: float | None = None
    fv_light_rmsd: float | None = None
    cdr_h1_rmsd: float | None = None
    cdr_h2_rmsd: float | None = None
    cdr_h3_rmsd: float | None = None
    cdr_l1_rmsd: float | None = None
    cdr_l2_rmsd: float | None = None
    cdr_l3_rmsd: float | None = None
    full_ab_rmsd: float | None = None

    @classmethod
    def from_residue_array(
        cls,
        predicted: ResidueArray,
        target: ResidueArray,
        heavy_chain: str | None = "H",
        light_chain: str | None = "L",
        atom_selector: Callable[[ResidueArray], Tensor] | Tensor | None = None,
    ):
        chains = []
        if heavy_chain is not None:
            chains.append(heavy_chain)

        if light_chain is not None:
            chains.append(light_chain)

        predicted, _, full_ab_rmsd = superimpose(
            target,
            predicted,
            residue_selector=ChainSelector(chains),
            atom_selector=atom_selector,
            rename_symmetric_atoms=True,
        )

        full_ab_rmsd = full_ab_rmsd.item()

        if heavy_chain is not None:
            fv_heavy_rmsd = rmsd(
                predicted,
                target,
                residue_selector=ChainSelector([heavy_chain]),
                atom_selector=atom_selector,
            ).item()
            heavy_cdr_rmsds = {
                f"cdr_h{i}_rmsd": rmsd(
                    predicted,
                    target,
                    residue_selector=CDRResidueSelector(
                        which_cdrs=[f"H{i}"],
                        heavy_chain=heavy_chain,
                        light_chain=light_chain,
                    ),
                    atom_selector=atom_selector,
                    rename_symmetric_atoms=False,
                ).item()
                for i in (1, 2, 3)
            }
        else:
            fv_heavy_rmsd = None
            heavy_cdr_rmsds = {}

        if light_chain is not None:
            fv_light_rmsd = rmsd(
                predicted,
                target,
                residue_selector=ChainSelector([light_chain]),
                atom_selector=atom_selector,
            ).item()
            light_cdr_rmsds = {
                f"cdr_l{i}_rmsd": rmsd(
                    predicted,
                    target,
                    residue_selector=CDRResidueSelector(
                        which_cdrs=[f"L{i}"],
                        heavy_chain=heavy_chain,
                        light_chain=light_chain,
                    ),
                    atom_selector=atom_selector,
                    rename_symmetric_atoms=False,
                ).item()
                for i in (1, 2, 3)
            }
        else:
            fv_light_rmsd = None
            light_cdr_rmsds = {}

        return cls(
            full_ab_rmsd=full_ab_rmsd,
            fv_heavy_rmsd=fv_heavy_rmsd,
            fv_light_rmsd=fv_light_rmsd,
            **heavy_cdr_rmsds,
            **light_cdr_rmsds,
        )
