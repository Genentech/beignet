import beignet
import torch


# def test_chebyshev_polynomial_roots():
#     torch.testing.assert_close(
#         beignet.chebyshev_polynomial_roots(
#             torch.tensor([1.0]),
#         ),
#         torch.tensor([]),
#     )
#
#     torch.testing.assert_close(
#         beignet.chebyshev_polynomial_roots(
#             torch.tensor([1.0, 2.0]),
#         ),
#         torch.tensor([-0.5]),
#     )
#
#     for i in range(2, 5):
#         torch.testing.assert_close(
#             beignet.trim_chebyshev_polynomial_coefficients(
#                 beignet.chebyshev_polynomial_roots(
#                     beignet.chebyshev_polynomial_from_roots(
#                         torch.linspace(-1, 1, i),
#                     )
#                 ),
#                 tol=0.000001,
#             ),
#             beignet.trim_chebyshev_polynomial_coefficients(
#                 torch.linspace(-1, 1, i),
#                 tol=0.000001,
#             ),
#         )
