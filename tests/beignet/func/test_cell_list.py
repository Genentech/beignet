import unittest
import torch
from torch.testing import assert_allclose
from beignet.func._molecular_dynamics._partition._cell_list import cell_list


class CellListTest(unittest.TestCase):
    def test_cell_list_emplace_2d(self):
        dtype = torch.float32
        box_size = torch.tensor([8.65, 8.0], dtype=torch.float32)
        cell_size = 1.0

        # Test particle positions
        R = torch.tensor(
            [[0.25, 0.25], [8.5, 1.95], [8.1, 1.5], [3.7, 7.9]], dtype=dtype
        )

        cell_fn = cell_list(box_size, cell_size)

        cell_list_instance = cell_fn.setup_fn(R)

        self.assertEqual(cell_list_instance.indexes.dtype, torch.int32)

        assert_allclose(R[0], cell_list_instance.positions_buffer[0, 0, 0])
        assert_allclose(R[1], cell_list_instance.positions_buffer[1, 7, 1])
        assert_allclose(R[2], cell_list_instance.positions_buffer[1, 7, 0])
        assert_allclose(R[3], cell_list_instance.positions_buffer[7, 3, 1])

        self.assertEqual(0, cell_list_instance.indexes[0, 0, 0])
        self.assertEqual(1, cell_list_instance.indexes[1, 7, 1])
        self.assertEqual(2, cell_list_instance.indexes[1, 7, 0])
        self.assertEqual(3, cell_list_instance.indexes[7, 3, 1])

        id_flat = cell_list_instance.indexes.view(-1)
        R_flat = cell_list_instance.positions_buffer.view(-1, 2)

        R_out = torch.zeros((5, 2), dtype=dtype)
        R_out[id_flat] = R_flat
        R_out = R_out[:-1]

        assert_allclose(R_out, R)

    def test_cell_list_random_emplace(self):
        dtype = torch.float32
        dim = 2  # Change this to 3 if you want to test for 3D
        particle_count = 10
        torch.manual_seed(1)

        box_size = 9.0
        cell_size = 1.0

        R = box_size * torch.rand((particle_count, dim), dtype=dtype)

        cell_fn = cell_list(torch.tensor([box_size] * dim, dtype=dtype), cell_size)
        cell_list_instance = cell_fn.setup_fn(R)

        id_flat = cell_list_instance.indexes.view(-1)
        R_flat = cell_list_instance.positions_buffer.view(-1, dim)
        R_out = torch.zeros((particle_count + 1, dim), dtype=dtype)
        R_out[id_flat] = R_flat
        R_out = R_out[:-1]

        assert_allclose(R_out, R)

    def test_cell_list_random_emplace_rect(self):
        dtype = torch.float32
        dim = 2  # Change this to 3 if you want to test for 3D
        particle_count = 10
        torch.manual_seed(1)

        box_size = (
            torch.tensor([9.0, 3.25], dtype=dtype)
            if dim == 2
            else torch.tensor([9.0, 3.0, 7.25], dtype=dtype)
        )
        cell_size = 1.0

        R = box_size * torch.rand((particle_count, dim), dtype=dtype)

        cell_fn = cell_list(box_size, cell_size)
        cell_list_instance = cell_fn.setup_fn(R)

        id_flat = cell_list_instance.indexes.view(-1)
        R_flat = cell_list_instance.positions_buffer.view(-1, dim)
        R_out = torch.zeros((particle_count + 1, dim), dtype=dtype)
        R_out[id_flat] = R_flat
        R_out = R_out[:-1]

        assert_allclose(R_out, R)

    def test_cell_list_random_emplace_side_data(self):
        dtype = torch.float32
        dim = 2  # Change this to 3 if you want to test for 3D
        particle_count = 10
        torch.manual_seed(1)

        box_size = (
            torch.tensor([9.0, 4.25], dtype=dtype)
            if dim == 2
            else torch.tensor([9.0, 4.0, 7.25], dtype=dtype)
        )
        cell_size = 1.23

        R = box_size * torch.rand((particle_count, dim), dtype=dtype)
        side_data_dim = 2
        side_data = torch.randn((particle_count, side_data_dim), dtype=dtype)

        cell_fn = cell_list(box_size, cell_size)
        cell_list_instance = cell_fn.setup_fn(R, side_data=side_data)

        id_flat = cell_list_instance.indexes.view(-1)
        R_flat = cell_list_instance.positions_buffer.view(-1, dim)
        R_out = torch.zeros((particle_count + 1, dim), dtype=dtype)
        R_out[id_flat] = R_flat
        R_out = R_out[:-1]

        side_data_flat = cell_list_instance.parameters["side_data"].view(
            -1, side_data_dim
        )
        side_data_out = torch.zeros((particle_count + 1, side_data_dim), dtype=dtype)
        side_data_out[id_flat] = side_data_flat
        side_data_out = side_data_out[:-1]

        assert_allclose(R_out, R)
        assert_allclose(side_data_out, side_data)
