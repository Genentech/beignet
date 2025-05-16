import einops
from torch import Tensor

from ._contact_matrix import contact_matrix
from ._residue_array import ResidueArray
from ._superimpose import rmsd, superimpose
from .selectors import (
    AndSelector,
    ChainSelector,
    InterfaceResidueSelector,
    PeptideBackboneSelector,
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
        selector_A=AndSelector([ChainSelector(receptor_chains), atom_thin_mask]),
        selector_B=AndSelector([ChainSelector(ligand_chains), atom_thin_mask]),
        radius_cutoff=5.0,
    )

    model_contacts = contact_matrix(
        model,
        selector_A=AndSelector([ChainSelector(receptor_chains), atom_thin_mask]),
        selector_B=AndSelector([ChainSelector(ligand_chains), atom_thin_mask]),
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
    rename_symmetric_atoms: bool = True,
) -> dict[str, Tensor]:
    _, _, interface_rmsd = superimpose(
        native,
        model,
        selector=AndSelector(
            [
                InterfaceResidueSelector(
                    ChainSelector(receptor_chains),
                    ChainSelector(ligand_chains),
                    radius_cutoff=radius_cutoff,
                ),
                PeptideBackboneSelector(include_oxygen=True),
            ]
        ),
        rename_symmetric_atoms=rename_symmetric_atoms,
    )

    return {"interface_rmsd": interface_rmsd}


def dockq_lrmsd_score(
    model: "ResidueArray",
    native: "ResidueArray",
    receptor_chains: list[str],
    ligand_chains: list[str],
    rename_symmetric_atoms: bool = True,
) -> dict[str, Tensor]:
    # align on receptor
    model, _, receptor_rmsd = superimpose(
        native,
        model,
        selector=AndSelector(
            [
                ChainSelector(receptor_chains),
                PeptideBackboneSelector(include_oxygen=True),
            ]
        ),
        rename_symmetric_atoms=rename_symmetric_atoms,
    )

    # rmsd on ligand
    ligand_rmsd = rmsd(
        model,
        native,
        selector=AndSelector(
            [
                ChainSelector(ligand_chains),
                PeptideBackboneSelector(include_oxygen=True),
            ]
        ),
    )

    return {"receptor_rmsd": receptor_rmsd, "ligand_rmsd": ligand_rmsd}


def f1(tp, fp, p):
    return 2 * tp / (tp + fp + p)


def dockq_formula(fnat, irms, lrms):
    return (
        fnat
        + 1 / (1 + (irms / 1.5) * (irms / 1.5))
        + 1 / (1 + (lrms / 8.5) * (lrms / 8.5))
    ) / 3


def dockq(
    model: "ResidueArray",
    native: "ResidueArray",
    receptor_chains: list[str],
    ligand_chains: list[str],
    rename_symmetric_atoms: bool = True,
) -> dict[str, Tensor]:
    contact = dockq_contact_score(
        model=model,
        native=native,
        receptor_chains=receptor_chains,
        ligand_chains=ligand_chains,
    )

    irmsd = dockq_irmsd_score(
        model=model,
        native=native,
        receptor_chains=receptor_chains,
        ligand_chains=ligand_chains,
        rename_symmetric_atoms=rename_symmetric_atoms,
    )

    lrmsd = dockq_lrmsd_score(
        model=model,
        native=native,
        receptor_chains=receptor_chains,
        ligand_chains=ligand_chains,
        rename_symmetric_atoms=rename_symmetric_atoms,
    )

    f1_value = f1(
        contact["shared_contacts"],
        contact["non_native_contacts"],
        contact["native_contacts"],
    )
    fnat = contact["shared_contacts"] / contact["native_contacts"].clamp(min=1)
    fnonnat = contact["non_native_contacts"] / contact["model_contacts"].clamp(min=1)

    dockq_value = dockq_formula(fnat, irmsd["interface_rmsd"], lrmsd["ligand_rmsd"])

    return {
        "DockQ": dockq_value,
        "f1": f1_value,
        "fnat": fnat,
        "fnonnat": fnonnat,
        **contact,
        **irmsd,
        **lrmsd,
    }
