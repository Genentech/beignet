import beignet
import torch


# def test_multiply_chebyshev_polynomial():
#     for j in range(5):
#         for k in range(5):
#             target = torch.zeros(j + k + 1)
#
#             target[abs(j + k)] = target[abs(j + k)] + 0.5
#             target[abs(j - k)] = target[abs(j - k)] + 0.5
#
#             input = torch.tensor([0.0] * j + [1.0])
#             other = torch.tensor([0.0] * k + [1.0])
#
#             torch.testing.assert_close(
#                 beignet.trim_chebyshev_polynomial_coefficients(
#                     beignet.multiply_chebyshev_polynomial(
#                         input,
#                         other,
#                     ),
#                     tol=0.000001,
#                 ),
#                 beignet.trim_chebyshev_polynomial_coefficients(
#                     target,
#                     tol=0.000001,
#                 ),
#             )
