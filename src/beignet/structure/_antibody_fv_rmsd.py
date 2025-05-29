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


def antibody_fv_rmsd(
    model: ResidueArray,
    native: ResidueArray,
    heavy_chain: str | None = "H",
    light_chain: str | None = "L",
    selector: Callable[[ResidueArray], Tensor] | Tensor | None = None,
    rename_symmetric_atoms: bool = True,
    include_cdr4: bool = False,
) -> dict[str, Tensor | None]:
    """Calculate RMSDs for antibody fv structures.

    This function superimposes the model structure onto the native structure based
    on the framework regions (FW) to calculate the full antibody RMSD.
    Then it superimposes the heavy and light chains separately
    to calculate their respective full and cdr RMSDs.
    See notes for definitions.

    Parameters
    ----------
    model : ResidueArray
        The model antibody structure.
    native : ResidueArray
        The native (reference) antibody structure.
    heavy_chain : str | None = "H"
        Identifier for the heavy chain.
    light_chain : str | None = "L"
        Identifier for the light chain.
    selector : Callable[[ResidueArray], Tensor] | Tensor | None = None
        An additional selector to be applied. This selector
        is combined with the internal selectors using an AND operation.
        For example, this can be used to select only C-alpha atoms.
    rename_symmetric_atoms : bool | None = None
        If True, symmetrically equivalent atoms (e.g., in ARG, ASP, GLU, PHE, TYR)
        will be renamed to minimize lddt during the initial framework-based
        superimposition. Default is True.
    include_cdr4: bool = False
        If True, include H4 and L4 cdrs

    Returns
    -------
    dict[str, Tensor or None]
        A dictionary containing the calculated RMSD values. Keys are:
        - "full_ab_rmsd": RMSD of the full antibody (heavy and light chains)
          after superimposition on the combined framework regions.
        - "heavy_chain_rmsd": RMSD of the heavy chain after superimposition
          on the heavy chain framework. None if `heavy_chain` is None.
        - "light_chain_rmsd": RMSD of the light chain after superimposition
          on the light chain framework. None if `light_chain` is None.
        - "cdr_h1_rmsd", "cdr_h2_rmsd", "cdr_h3_rmsd", "cdr_h4_rmsd": RMSD for
          each heavy chain CDR (H1, H2, H3, H4) after superimposition on the
          heavy chain framework.
        - "cdr_l1_rmsd", "cdr_l2_rmsd", "cdr_l3_rmsd", "cdr_l4_rmsd": RMSD for
          each light chain CDR (L1, L2, L3, L4) after superimposition on the
          light chain framework.

    Notes
    -----
    The function defines Complementarity Determining Regions (CDRs) and
    Framework Regions (FW) as follows:
    - HCDR: CDRs of the heavy chain (H1, H2, H3, H4?)
    - HFW: Framework regions of the heavy chain (parts of the heavy chain that are not HCDRs).
    - LCDR: CDRs of the light chain (L1, L2, L3, L4?)
    - LFW: Framework regions of the light chain (parts of the light chain that are not LCDRs).
    - FW: Combined framework regions of both heavy and light chains.
    """

    heavy_cdrs = [f"H{i}" for i in (1, 2, 3)]
    if include_cdr4:
        heavy_cdrs = [*heavy_cdrs, "H4"]

    light_cdrs = [f"L{i}" for i in (1, 2, 3)]
    if include_cdr4:
        light_cdrs = [*light_cdrs, "L4"]

    HCDR = CDRResidueSelector(
        heavy_cdrs, heavy_chain=heavy_chain, light_chain=light_chain
    )
    HFW = AndSelector([ChainSelector([heavy_chain]), NotSelector(HCDR)])

    LCDR = CDRResidueSelector(
        light_cdrs, heavy_chain=heavy_chain, light_chain=light_chain
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
            f"cdr_{cdr.lower()}_rmsd": rmsd(
                model,
                native,
                selector=AndSelector(
                    [
                        CDRResidueSelector(
                            which_cdrs=[cdr],
                            heavy_chain=heavy_chain,
                            light_chain=light_chain,
                        ),
                        selector,
                    ]
                ),
            )
            for cdr in heavy_cdrs
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
            f"cdr_{cdr.lower()}_rmsd": rmsd(
                model,
                native,
                selector=AndSelector(
                    [
                        CDRResidueSelector(
                            which_cdrs=[cdr],
                            heavy_chain=heavy_chain,
                            light_chain=light_chain,
                        ),
                        selector,
                    ]
                ),
            )
            for cdr in light_cdrs
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
