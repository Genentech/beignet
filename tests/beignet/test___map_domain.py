import beignet.polynomial
import torch


def test__map_domain():
    torch.testing.assert_close(
        beignet.polynomial._map_domain(
            torch.tensor([0, 4]),
            torch.tensor([0, 4]),
            torch.tensor([1, 3]),
        ),
        torch.tensor([1, 3], dtype=torch.float32),
    )

    torch.testing.assert_close(
        beignet.polynomial._map_domain(
            torch.tensor([0 - 1j, 2 + 1j]),
            torch.tensor([0 - 1j, 2 + 1j]),
            torch.tensor([-2, 2]),
        ),
        torch.tensor([-2, 2], dtype=torch.complex64),
    )

    torch.testing.assert_close(
        beignet.polynomial._map_domain(
            torch.tensor([[0, 4], [0, 4]]),
            torch.tensor([0, 4]),
            torch.tensor([1, 3]),
        ),
        torch.tensor([[1, 3], [1, 3]], dtype=torch.float32),
    )
