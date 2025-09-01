# import beignet
# import hypothesis.strategies
# import numpy
# import torch
# from scipy.spatial.transform import Rotation, Slerp
#
# def test_slerp():
#     # t = 0
#     torch.testing.assert_close(
#         beignet.quaternion_slerp(
#             torch.tensor([+0.00000]),
#             torch.tensor([+0.00000, +1.00000]),
#             torch.tensor(
#                 [
#                     [+1.00000, +0.00000, +0.00000, +0.00000],
#                     [+0.00000, +1.00000, +0.00000, +0.00000],
#                 ]
#             ),
#         ),
#         torch.tensor([[+1.00000, +0.00000, +0.00000, +0.00000]]),
#     )
#
#     # t = 1
#     torch.testing.assert_close(
#         beignet.quaternion_slerp(
#             torch.tensor([+1.00000]),
#             torch.tensor([+0.00000, +1.00000]),
#             torch.tensor(
#                 [
#                     [+1.00000, +0.00000, +0.00000, +0.00000],
#                     [+0.00000, +1.00000, +0.00000, +0.00000],
#                 ]
#             ),
#         ),
#         torch.tensor([[+0.00000, +1.00000, +0.00000, +0.00000]]),
#     )
#
#     # SMALL (ACUTE) ANGLE BETWEEN QUATERNIONS
#     torch.testing.assert_close(
#         beignet.quaternion_slerp(
#             torch.tensor([+0.50000]),
#             torch.tensor([+0.00000, +1.00000]),
#             torch.tensor(
#                 [
#                     [+1.00000, +0.00000, +0.00000, +0.00000],
#                     [+0.70710, +0.70710, +0.00000, +0.00000],
#                 ],
#             ),
#         ),
#         torch.reshape(
#             torch.tensor([+0.92388, +0.38268, +0.00000, +0.00000]),
#             [1, -1],
#         ),
#     )
#
#     # LARGE (OBTUSE) ANGLE BETWEEN QUATERNIONS
#     torch.testing.assert_close(
#         beignet.quaternion_slerp(
#             torch.tensor([+0.50000]),
#             torch.tensor([+0.00000, +1.00000]),
#             torch.tensor(
#                 [
#                     [+1.00000, +0.00000, +0.00000, +0.00000],
#                     [-1.00000, +0.00000, +0.00000, +0.00000],
#                 ]
#             ),
#         ),
#         torch.reshape(
#             torch.tensor([+1.00000, +0.00000, +0.00000, +0.00000]),
#             [1, -1],
#         ),
#     )
#
#
# @hypothesis.strategies.composite
# def slerp_parameters(f):
#     n = f(
#         hypothesis.strategies.integers(
#             min_value=2,
#             max_value=8,
#         ),
#     )
#
#     times = numpy.sort(
#         f(
#             hypothesis.strategies.lists(
#                 hypothesis.strategies.floats(
#                     allow_infinity=False,
#                     allow_nan=False,
#                 ),
#                 min_size=n,
#                 max_size=n,
#                 unique=True,
#             ),
#         ),
#     )
#
#     min_value = numpy.min(times)
#     max_value = numpy.max(times)
#
#     input = numpy.sort(
#         f(
#             hypothesis.strategies.lists(
#                 hypothesis.strategies.floats(
#                     min_value=min_value,
#                     max_value=max_value,
#                 ),
#                 min_size=1,
#                 max_size=8,
#                 unique=True,
#             ),
#         ),
#     )
#
#     rotations = f(
#         hypothesis.strategies.lists(
#             hypothesis.strategies.lists(
#                 hypothesis.strategies.floats(
#                     numpy.finfo(numpy.float32).eps,
#                     1.0,
#                 ),
#                 min_size=4,
#                 max_size=4,
#             ),
#             min_size=n,
#             max_size=n,
#         ),
#     )
#
#     rotations = Rotation.from_quat(rotations)
#
#     return [
#         [
#             torch.from_numpy(input),
#             torch.from_numpy(times),
#             torch.from_numpy(rotations.as_quat(canonical=True)),
#         ],
#         torch.from_numpy(
#             Slerp(times, rotations)(input).as_quat(canonical=True),
#         ),
#     ]
#
#
# @hypothesis.given(slerp_parameters())
# def test_slerp_properties(data):
#     parameters, expected_rotations = data
#
#     torch.testing.assert_close(
#         beignet.quaternion_slerp(*parameters), expected_rotations
#     )
