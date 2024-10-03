import math

import beignet
import torch


# def test_polynomial_from_roots():
#     coefficients = [
#         torch.tensor([1.0]),
#         torch.tensor([0.0, 1]),
#         torch.tensor([-1.0, 0, 2]),
#         torch.tensor([0.0, -3, 0, 4]),
#         torch.tensor([1.0, 0, -8, 0, 8]),
#         torch.tensor([0.0, 5, 0, -20, 0, 16]),
#         torch.tensor([-1.0, 0, 18, 0, -48, 0, 32]),
#         torch.tensor([0.0, -7, 0, 56, 0, -112, 0, 64]),
#         torch.tensor([1.0, 0, -32, 0, 160, 0, -256, 0, 128]),
#         torch.tensor([0.0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
#     ]
#
#     torch.testing.assert_close(
#         beignet.trim_polynomial_coefficients(
#             beignet.polynomial_from_roots(
#                 torch.tensor([]),
#             ),
#             tol=0.000001,
#         ),
#         torch.tensor([1.0]),
#     )
#
#     for index in range(1, 5):
#         input = torch.linspace(-math.pi, 0.0, 2 * index + 1)
#
#         input = input[1::2]
#
#         input = torch.cos(input)
#
#         output = beignet.polynomial_from_roots(input) * 2 ** (index - 1)
#
#         torch.testing.assert_close(
#             beignet.trim_polynomial_coefficients(
#                 output,
#                 tol=0.000001,
#             ),
#             beignet.trim_polynomial_coefficients(
#                 coefficients[index],
#                 tol=0.000001,
#             ),
#         )
