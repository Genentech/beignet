# import beignet
# import pytest
# import torch
#
#
# def test_differentiate_laguerre_polynomial():
#     with pytest.raises(TypeError):
#         beignet.differentiate_laguerre_polynomial(
#             torch.tensor([0]),
#             order=0.5,
#         )
#
#     with pytest.raises(ValueError):
#         beignet.differentiate_laguerre_polynomial(
#             torch.tensor([0]),
#             order=-1,
#         )
#
#     for i in range(5):
#         torch.testing.assert_close(
#             beignet.trim_laguerre_polynomial_coefficients(
#                 beignet.differentiate_laguerre_polynomial(
#                     torch.tensor([0.0] * i + [1.0]),
#                     order=0,
#                 ),
#                 tol=0.000001,
#             ),
#             beignet.trim_laguerre_polynomial_coefficients(
#                 torch.tensor([0.0] * i + [1.0]),
#                 tol=0.000001,
#             ),
#         )
#
#     for i in range(5):
#         for j in range(2, 5):
#             torch.testing.assert_close(
#                 beignet.trim_laguerre_polynomial_coefficients(
#                     beignet.differentiate_laguerre_polynomial(
#                         beignet.integrate_laguerre_polynomial(
#                             torch.tensor([0.0] * i + [1.0]),
#                             order=j,
#                         ),
#                         order=j,
#                     ),
#                     tol=0.000001,
#                 ),
#                 beignet.trim_laguerre_polynomial_coefficients(
#                     torch.tensor([0.0] * i + [1.0]),
#                     tol=0.000001,
#                 ),
#             )
#
#     for i in range(5):
#         for j in range(2, 5):
#             torch.testing.assert_close(
#                 beignet.trim_laguerre_polynomial_coefficients(
#                     beignet.differentiate_laguerre_polynomial(
#                         beignet.integrate_laguerre_polynomial(
#                             torch.tensor([0.0] * i + [1.0]),
#                             order=j,
#                             scale=2,
#                         ),
#                         order=j,
#                         scale=0.5,
#                     ),
#                     tol=0.000001,
#                 ),
#                 beignet.trim_laguerre_polynomial_coefficients(
#                     torch.tensor([0.0] * i + [1.0]),
#                     tol=0.000001,
#                 ),
#             )
#
#     # c2d = torch.rand(3, 4)
#
#     # torch.testing.assert_close(
#     #     beignet.lagder(c2d, axis=0),
#     #     torch.vstack([beignet.lagder(c) for c in c2d.T]).T,
#     # )
#
#     # torch.testing.assert_close(
#     #     beignet.lagder(
#     #         c2d,
#     #         axis=1,
#     #     ),
#     #     torch.vstack([beignet.lagder(c) for c in c2d]),
#     # )
