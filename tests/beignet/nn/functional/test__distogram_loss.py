import beignet.nn.functional
import torch


def test_distogram_loss():
    torch.testing.assert_close(
        beignet.nn.functional.distogram_loss(
            torch.zeros([1, 8, 8, 8]),
            torch.zeros([1, 8, 8, 8]),
            torch.zeros([1, 8, 8, 8]),
            steps=8,
        ),
        torch.tensor(0.0),
    )
