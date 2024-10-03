import math

import beignet
import torch


# def test_chebyshev_polynomial_from_roots():
#     torch.testing.assert_close(
#         beignet.trim_chebyshev_polynomial_coefficients(
#             beignet.chebyshev_polynomial_from_roots(
#                 torch.tensor([]),
#             ),
#             tol=0.000001,
#         ),
#         torch.tensor([1.0]),
#     )
#
#     for index in range(1, 5):
#         input = beignet.chebyshev_polynomial_from_roots(
#             torch.cos(torch.linspace(-math.pi, 0.0, 2 * index + 1)[1::2]),
#         )
#
#         input = input * 2 ** (index - 1)
#
#         torch.testing.assert_close(
#             beignet.trim_chebyshev_polynomial_coefficients(
#                 input,
#                 tol=0.000001,
#             ),
#             beignet.trim_chebyshev_polynomial_coefficients(
#                 torch.tensor([0.0] * index + [1.0]),
#                 tol=0.000001,
#             ),
#         )
