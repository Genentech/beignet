import beignet.nn.functional
import torch


def test_torsion_angle_loss():
    input = torch.ones([1, 1, 7, 2])

    target = torch.zeros([1, 1, 7, 2]), torch.zeros([1, 1, 7, 2])

    output = beignet.nn.functional.torsion_angle_loss(input, target)

    torch.testing.assert_close(
        output,
        torch.tensor([1.0]),
        rtol=0.01,
        atol=0.01,
    )
