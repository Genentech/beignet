# import beignet
# import torch
#
#
# def test_gauss_laguerre_quadrature():
#     x, w = beignet.gauss_laguerre_quadrature(100)
#
#     v = beignet.lagvander(x, 99)
#     vv = (v.T * w) @ v
#     vd = 1 / torch.sqrt(vv.diagonal())
#     vv = vd[:, None] * vv * vd
#     torch.testing.assert_close(
#         vv,
#         torch.eye(100),
#     )
#
#     target = 1.0
#     torch.testing.assert_close(w.sum(), target)
