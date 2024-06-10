import beignet
import torch


def test_root():
    c = torch.linspace(2, 100, 1001, dtype=torch.float64)

    output, _ = beignet.root(
        lambda x: x**2 - c,
        torch.sqrt(c) - 1.1,
        torch.sqrt(c) + 1.0,
    )

    torch.testing.assert_close(output, torch.sqrt(c))
