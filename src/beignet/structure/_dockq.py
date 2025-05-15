import einops
from torch import Tensor

from ._residue_array import ResidueArray
from ._superimpose import rmsd, superimpose
from ._contact_matrix import contact_matrix
from .selectors import (
    AndSelector,
    AtomNameSelector,
    ChainSelector,
    InterfaceResidueSelector,
)


def dockq_contact_score(
    model: ResidueArray,
    native: ResidueArray,
    receptor_chains: list[str],
    ligand_chains: list[str],
) -> dict[str, Tensor]:
    # only look at atoms present in both model and native
    atom_thin_mask = model.atom_thin_mask & native.atom_thin_mask

    native_contacts = contact_matrix(
        native,
        selector_A=AndSelector(ChainSelector(receptor_chains), atom_thin_mask),
        selector_B=AndSelector(ChainSelector(ligand_chains), atom_thin_mask),
        radius_cutoff=5.0,
    )

    model_contacts = contact_matrix(
        model,
        selector_A=AndSelector(ChainSelector(receptor_chains), atom_thin_mask),
        selector_B=AndSelector(ChainSelector(ligand_chains), atom_thin_mask),
        radius_cutoff=5.0,
    )

    shared_contacts = model_contacts & native_contacts
    non_native_contacts = model_contacts & ~native_contacts

    native_contacts = einops.reduce(native_contacts, "... l lp -> ...", "sum") // 2
    model_contacts = einops.reduce(model_contacts, "... l lp -> ...", "sum") // 2
    shared_contacts = einops.reduce(shared_contacts, "... l lp -> ...", "sum") // 2
    non_native_contacts = (
        einops.reduce(non_native_contacts, "... l lp -> ...", "sum") // 2
    )

    return {
        "native_contacts": native_contacts,
        "model_contacts": model_contacts,
        "shared_contacts": shared_contacts,
        "non_native_contacts": non_native_contacts,
    }


def dockq_irmsd_score(
    model: "ResidueArray",
    native: "ResidueArray",
    receptor_chains: list[str],
    ligand_chains: list[str],
    radius_cutoff: float = 10.0,
) -> dict[str, Tensor]:
    _, _, interface_rmsd = superimpose(
        native,
        model,
        selector=AndSelector(
            InterfaceResidueSelector(
                ChainSelector(receptor_chains),
                ChainSelector(ligand_chains),
                radius_cutoff=radius_cutoff,
            ),
            AtomNameSelector(["CA", "C", "N", "O"]),
        ),
        rename_symmetric_atoms=False,
    )

    return {"interface_rmsd": interface_rmsd}


def dockq_lrmsd_score(
    model: "ResidueArray",
    native: "ResidueArray",
    receptor_chains: list[str],
    ligand_chains: list[str],
) -> dict[str, Tensor]:
    # align on receptor
    model, _, receptor_rmsd = superimpose(
        native,
        model,
        selector=AndSelector(
            ChainSelector(receptor_chains),
            AtomNameSelector(["CA", "C", "N", "O"]),
        ),
    )

    # rmsd on ligand
    ligand_rmsd = rmsd(
        model,
        native,
        selector=AndSelector(
            ChainSelector(ligand_chains),
            AtomNameSelector(["CA", "C", "N", "O"]),
        ),
    )

    return {"receptor_rmsd": receptor_rmsd, "ligand_rmsd": ligand_rmsd}
