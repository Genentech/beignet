import beignet.polynomial
import beignet.polynomial.__map_parameters
import torch


def test__map_parameters():
    torch.testing.assert_close(
        beignet.polynomial.__map_parameters._map_parameters([0, 4], [1, 3]), [1, 0.5]
    )
    torch.testing.assert_close(
        beignet.polynomial.__map_parameters._map_parameters([0 - 1j, 2 + 1j], [-2, 2]),
        [-1 + 1j, 1 - 1j],
    )
