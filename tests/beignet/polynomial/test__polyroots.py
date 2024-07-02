import torch
from beignet.polynomial import polyfromroots, polyroots, polytrim


def test_polyroots():
    torch.testing.assert_close(
        polyroots(torch.tensor([1.0])),
        torch.tensor([]),
    )

    torch.testing.assert_close(
        polyroots(torch.tensor([1.0, 2.0])),
        torch.tensor([-0.5]),
    )

    for index in range(2, 5):
        input = torch.linspace(-1, 1, index)

        torch.testing.assert_close(
            polytrim(
                polyroots(
                    polyfromroots(
                        input,
                    ),
                ),
                tol=0.000001,
            ),
            polytrim(
                input,
                tol=0.000001,
            ),
        )
