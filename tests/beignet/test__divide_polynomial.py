import beignet
import torch


# def test_divide_polynomial():
#     quotient, remainder = beignet.divide_polynomial(
#         torch.tensor([2.0]),
#         torch.tensor([2.0]),
#     )
#
#     torch.testing.assert_close(
#         quotient,
#         torch.tensor([1.0]),
#     )
#
#     torch.testing.assert_close(
#         remainder,
#         torch.tensor([0.0]),
#     )
#
#     quotient, remainder = beignet.divide_polynomial(
#         torch.tensor([2.0, 2.0]),
#         torch.tensor([2.0]),
#     )
#
#     torch.testing.assert_close(
#         quotient,
#         torch.tensor([1.0, 1.0]),
#     )
#
#     torch.testing.assert_close(
#         remainder,
#         torch.tensor([0.0]),
#     )
#
#     for j in range(5):
#         for k in range(5):
#             input = torch.tensor([0.0] * j + [1.0, 2.0])
#             other = torch.tensor([0.0] * k + [1.0, 2.0])
#
#             quotient, remainder = beignet.divide_polynomial(
#                 beignet.add_polynomial(
#                     input,
#                     other,
#                 ),
#                 input,
#             )
#
#             torch.testing.assert_close(
#                 beignet.trim_polynomial_coefficients(
#                     beignet.add_polynomial(
#                         beignet.multiply_polynomial(
#                             quotient,
#                             input,
#                         ),
#                         remainder,
#                     ),
#                     tol=0.000001,
#                 ),
#                 beignet.trim_polynomial_coefficients(
#                     beignet.add_polynomial(
#                         input,
#                         other,
#                     ),
#                     tol=0.000001,
#                 ),
#             )
