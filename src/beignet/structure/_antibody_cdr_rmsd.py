from typing import Callable

from torch import Tensor

from ._residue_array import ResidueArray
from ._superimpose import rmsd, superimpose
from .selectors import (
    AndSelector,
    CDRResidueSelector,
    ChainSelector,
    NotSelector,
    OrSelector,
)


def antibody_cdr_rmsd(
    model: ResidueArray,
    native: ResidueArray,
    heavy_chain: str | None = "H",
    light_chain: str | None = "L",
    selector: Callable[[ResidueArray], Tensor] | Tensor | None = None,
    rename_symmetric_atoms: bool = True,
) -> dict[str, Tensor | None]:
    HCDR = CDRResidueSelector(
        ["H1", "H2", "H3", "H4"], heavy_chain=heavy_chain, light_chain=light_chain
    )
    HFW = AndSelector([ChainSelector([heavy_chain]), NotSelector(HCDR)])

    LCDR = CDRResidueSelector(
        ["L1", "L2", "L3", "L4"], heavy_chain=heavy_chain, light_chain=light_chain
    )
    LFW = AndSelector([ChainSelector([light_chain]), NotSelector(LCDR)])

    FW = OrSelector([HFW, LFW])

    model, _, _ = superimpose(
        native,
        model,
        selector=AndSelector([FW, selector]),
        rename_symmetric_atoms=rename_symmetric_atoms,
    )

    full_ab_rmsd = rmsd(
        model,
        native,
        selector=AndSelector([ChainSelector([heavy_chain, light_chain]), selector]),
    )

    if heavy_chain is not None:
        model, _, _ = superimpose(
            native,
            model,
            selector=AndSelector([HFW, selector]),
            rename_symmetric_atoms=False,
        )

        heavy_rmsd = rmsd(
            model,
            native,
            selector=AndSelector([ChainSelector([heavy_chain]), selector]),
        )

        heavy_cdr_rmsds = {
            f"cdr_h{i}_rmsd": rmsd(
                model,
                native,
                selector=AndSelector(
                    [
                        CDRResidueSelector(
                            which_cdrs=[f"H{i}"],
                            heavy_chain=heavy_chain,
                            light_chain=light_chain,
                        ),
                        selector,
                    ]
                ),
                rename_symmetric_atoms=False,
            )
            for i in (1, 2, 3, 4)
        }
    else:
        heavy_rmsd = None
        heavy_cdr_rmsds = {}

    if light_chain is not None:
        model, _, _ = superimpose(
            native,
            model,
            selector=AndSelector([LFW, selector]),
            rename_symmetric_atoms=False,
        )

        light_rmsd = rmsd(
            model,
            native,
            selector=AndSelector([ChainSelector([light_chain]), selector]),
        )
        light_cdr_rmsds = {
            f"cdr_l{i}_rmsd": rmsd(
                model,
                native,
                selector=AndSelector(
                    [
                        CDRResidueSelector(
                            which_cdrs=[f"L{i}"],
                            heavy_chain=heavy_chain,
                            light_chain=light_chain,
                        ),
                        selector,
                    ]
                ),
                rename_symmetric_atoms=False,
            )
            for i in (1, 2, 3, 4)
        }
    else:
        light_rmsd = None
        light_cdr_rmsds = {}

    return {
        "full_ab_rmsd": full_ab_rmsd,
        "heavy_chain_rmsd": heavy_rmsd,
        "light_chain_rmsd": light_rmsd,
        **heavy_cdr_rmsds,
        **light_cdr_rmsds,
    }
