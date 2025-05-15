from typing import Callable

from torch import Tensor

from ._residue_array import ResidueArray
from ._superimpose import rmsd, superimpose
from .selectors import (
    AndSelector,
    CDRResidueSelector,
    ChainSelector,
)


def antibody_cdr_rmsd(
    model: ResidueArray,
    native: ResidueArray,
    heavy_chain: str | None = "H",
    light_chain: str | None = "L",
    selector: Callable[[ResidueArray], Tensor] | Tensor | None = None,
) -> dict[str, Tensor | None]:
    chains = []
    if heavy_chain is not None:
        chains.append(heavy_chain)

    if light_chain is not None:
        chains.append(light_chain)

    model, _, full_ab_rmsd = superimpose(
        native,
        model,
        selector=AndSelector([ChainSelector(chains), selector]),
        rename_symmetric_atoms=True,
    )

    if heavy_chain is not None:
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
            for i in (1, 2, 3)
        }
    else:
        heavy_rmsd = None
        heavy_cdr_rmsds = {}

    if light_chain is not None:
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
            for i in (1, 2, 3)
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
