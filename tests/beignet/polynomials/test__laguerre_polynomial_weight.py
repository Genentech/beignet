# import beignet
# import torch
#
#
# def test_laguerre_polynomial_weight():
#     torch.testing.assert_close(
#         beignet.laguerre_polynomial_weight(
#             torch.linspace(0, 10, 11),
#         ),
#         torch.exp(-torch.linspace(0, 10, 11)),
#     )
