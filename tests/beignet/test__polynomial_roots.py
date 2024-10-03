import beignet
import torch


# def test_polynomial_roots():
#     torch.testing.assert_close(
#         beignet.polynomial_roots(torch.tensor([1.0])),
#         torch.tensor([]),
#     )
#
#     torch.testing.assert_close(
#         beignet.polynomial_roots(torch.tensor([1.0, 2.0])),
#         torch.tensor([-0.5]),
#     )
#
#     for index in range(2, 5):
#         input = torch.linspace(-1, 1, index)
#
#         torch.testing.assert_close(
#             beignet.trim_polynomial_coefficients(
#                 beignet.polynomial_roots(
#                     beignet.polynomial_from_roots(
#                         input,
#                     ),
#                 ),
#                 tol=0.000001,
#             ),
#             beignet.trim_polynomial_coefficients(
#                 input,
#                 tol=0.000001,
#             ),
#         )
