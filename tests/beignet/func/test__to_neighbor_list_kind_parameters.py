# import unittest
# import torch
#
# from beignet.func._interact import _ParameterTree, _ParameterTreeKind, \
#     _to_neighbor_list_kind_parameters
# from beignet.func._partition import _NeighborListFormat
#
#
# class TestToNeighborListKindParameters(unittest.TestCase):
#
#     def setUp(self):
#         # Setup common test data
#         self.indexes = torch.tensor([[0, 1], [1, 2], [2, 3]])
#         self.kinds = torch.tensor([0, 1, 2, 3])
#         self.scalar_parameter = torch.tensor(0.5)
#         self.matrix_parameter = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
#         self.float_parameter = 0.5
#         self.parameter_tree = _ParameterTree(kind=_ParameterTreeKind.KINDS, tree={'param': torch.tensor([[0.1, 0.2], [0.3, 0.4]])})
#
#     def test_scalar_tensor_parameter(self):
#         result = _to_neighbor_list_kind_parameters(_NeighborListFormat.DENSE, self.indexes, self.kinds, self.scalar_parameter)
#
#         self.assertTrue(torch.equal(result, self.scalar_parameter))

    # def test_matrix_tensor_parameter_dense(self):
    #     result = _to_neighbor_list_kind_parameters(_NeighborListFormat.DENSE, self.indexes, self.kinds, self.matrix_parameter)
    #
    #     expected = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]]])
    #
    #     self.assertTrue(torch.equal(result, expected))
#
    # def test_matrix_tensor_parameter_sparse(self):
    #     result = _to_neighbor_list_kind_parameters(_NeighborListFormat.SPARSE, self.indexes, self.kinds, self.matrix_parameter)
    #
    #     expected = torch.tensor([0.2, 0.4, 0.4])
    #
    #     # self.assertTrue(torch.equal(result, expected))
#
#     def test_float_parameter(self):
#         result = _to_neighbor_list_kind_parameters(_NeighborListFormat.DENSE, self.indexes, self.kinds, self.float_parameter)
#         self.assertEqual(result, self.float_parameter)
#
#     def test_parameter_tree_kinds_dense(self):
#         result = _to_neighbor_list_kind_parameters(_NeighborListFormat.DENSE, self.indexes, self.kinds, self.parameter_tree)
#         expected = {'param': torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]]])}
#         self.assertTrue(torch.equal(result['param'], expected['param']))
#
#     def test_parameter_tree_kinds_sparse(self):
#         result = _to_neighbor_list_kind_parameters(_NeighborListFormat.SPARSE, self.indexes, self.kinds, self.parameter_tree)
#         expected = {'param': torch.tensor([0.2, 0.4, 0.4])}
#         self.assertTrue(torch.equal(result['param'], expected['param']))
#
#     def test_parameter_tree_space(self):
#         parameter_tree_space = _ParameterTree(kind=_ParameterTreeKind.SPACE, tree={'param': torch.tensor([1.0, 2.0])})
#         result = _to_neighbor_list_kind_parameters(_NeighborListFormat.DENSE, self.indexes, self.kinds, parameter_tree_space)
#         self.assertEqual(result, parameter_tree_space.tree)
#
#     def test_invalid_parameter_type(self):
#         with self.assertRaises(ValueError):
#             _to_neighbor_list_kind_parameters(_NeighborListFormat.DENSE, self.indexes, self.kinds, "invalid_parameter")
#
#     def test_invalid_tensor_shape(self):
#         invalid_tensor = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
#         with self.assertRaises(ValueError):
#             _to_neighbor_list_kind_parameters(_NeighborListFormat.DENSE, self.indexes, self.kinds, invalid_tensor)
