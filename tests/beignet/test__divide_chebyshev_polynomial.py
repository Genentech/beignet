import beignet
import torch


# def test_divide_chebyshev_polynomial():
#     for j in range(5):
#         for k in range(5):
#             input = torch.tensor([0.0] * j + [1.0])
#             other = torch.tensor([0.0] * k + [1.0])
#
#             quotient, remainder = beignet.divide_chebyshev_polynomial(
#                 beignet.add_chebyshev_polynomial(
#                     input,
#                     other,
#                 ),
#                 input,
#             )
#
#             torch.testing.assert_close(
#                 beignet.trim_chebyshev_polynomial_coefficients(
#                     beignet.add_chebyshev_polynomial(
#                         beignet.multiply_chebyshev_polynomial(
#                             quotient,
#                             input,
#                         ),
#                         remainder,
#                     ),
#                     tol=0.000001,
#                 ),
#                 beignet.trim_chebyshev_polynomial_coefficients(
#                     beignet.add_chebyshev_polynomial(
#                         input,
#                         other,
#                     ),
#                     tol=0.000001,
#                 ),
#             )
