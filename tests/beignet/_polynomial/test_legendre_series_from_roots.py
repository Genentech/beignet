import beignet.polynomial
import torch.testing


def test_legendre_series_from_roots():
    torch.testing.assert_close(
        beignet.polynomial.trim_legendre_series(
            beignet.polynomial.legendre_series_from_roots(
                torch.tensor([]),
            ),
            tolerance=0.000001,
        ),
        torch.tensor([1]),
    )

    # for index in range(1, 5):
    #     roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * index + 1)[1::2])
    #     pol = beignet.polynomial.legfromroots(roots)
    #     res = beignet.polynomial.evaluate_legendre_series_1d(roots, pol)
    #     tgt = 0
    #     assert len(pol) == index + 1
    #     torch.testing.assert_close(
    #         beignet.polynomial.legendre_series_to_power_series(pol)[-1], 1
    #     )
    #     torch.testing.assert_close(
    #         res,
    #         tgt,
    #     )
