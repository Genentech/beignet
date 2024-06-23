import beignet.polynomial
import torch.testing


def test__trim_sequence():
    for index in range(5):
        x = [0.0] * index

        torch.testing.assert_close(
            beignet.polynomial._trim_sequence(
                torch.tensor([1.0, *x]),
            ),
            torch.tensor([1.0]),
        )

    for x in [[], torch.tensor([], dtype=torch.int32)]:
        torch.testing.assert_close(
            beignet.polynomial._trim_sequence(x),
            x,
        )
