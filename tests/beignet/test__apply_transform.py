# import torch
# import torch.func
# from beignet import apply_transform
# from torch import Tensor
#
#
# def test_apply_transform():
#     input = torch.randn([32, 3])
#
#     transform = torch.randn([3, 3])
#
#     def f(r: Tensor) -> Tensor:
#         return torch.sum(r**2)
#
#     def g(r: Tensor, t: Tensor) -> Tensor:
#         return torch.sum(apply_transform(r, t) ** 2)
#
#     torch.testing.assert_close(
#         torch.func.grad(f)(apply_transform(input, transform)),
#         torch.func.grad(g, 0)(input, transform),
#     )
