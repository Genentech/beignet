# import beignet
# import torch
#
#
# def test_gauss_legendre_quadrature():
#     x, w = beignet.gauss_legendre_quadrature(100)
#
#     v = beignet.legvander(
#         x,
#         degree=torch.tensor([99]),
#     )
#
#     vv = (v.T * w) @ v
#
#     vd = 1 / torch.sqrt(vv.diagonal())
#     vv = vd[:, None] * vv * vd
#
#     torch.testing.assert_close(
#         vv,
#         torch.eye(100),
#     )
#
#     torch.testing.assert_close(w.sum(), 2.0)
