import beignet.polynomial
import beignet.polynomial.__map_parameters
import torch


def test__map_parameters():
    torch.testing.assert_close(
        beignet.polynomial._map_parameters(
            torch.tensor([0, 4]),
            torch.tensor([1, 3]),
        ),
        torch.tensor([1, 0.5]),
    )

    torch.testing.assert_close(
        beignet.polynomial._map_parameters(
            torch.tensor([+0 - 1j, +2 + 1j]),
            torch.tensor([-2 + 0j, +2 + 0j]),
        ),
        torch.tensor([-1 + 1j, +1 - 1j]),
    )
