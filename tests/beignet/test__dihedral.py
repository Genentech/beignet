import biotite.structure
import torch

from beignet import dihedral_angle


def test_dihedral():
    x = torch.randn(10, 4, 3)

    a, b, c, d = torch.unbind(x, dim=-2)

    phi_ref = biotite.structure.dihedral(a.numpy(), b.numpy(), c.numpy(), d.numpy())

    phi = dihedral_angle(x)

    print(f"{(phi - phi_ref).abs().max()=}")

    torch.testing.assert_close(phi, torch.as_tensor(phi_ref))


def test_dihedral_no_batch():
    x = torch.randn(4, 3)

    a, b, c, d = torch.unbind(x, dim=-2)

    phi_ref = biotite.structure.dihedral(a.numpy(), b.numpy(), c.numpy(), d.numpy())

    phi = dihedral_angle(x)

    print(f"{(phi - phi_ref).abs().max()=}")

    torch.testing.assert_close(phi, torch.as_tensor(phi_ref))
