import os

import torch

import beignet


class BenchKruskal:
    params = ([10, 50, 100], [0.1, 0.5, 1.0], [torch.float32, torch.float64])
    param_names = ["num_nodes", "density", "dtype"]

    def setup(self, num_nodes, density, dtype):
        torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 0x00B1)))

        # Generate random sparse graph
        num_possible_edges = num_nodes * (num_nodes - 1) // 2
        num_edges = max(num_nodes - 1, int(num_possible_edges * density))

        # Generate random edges
        rows = []
        cols = []
        vals = []

        # First create a tree to ensure connectivity
        for i in range(1, num_nodes):
            j = torch.randint(0, i, (1,)).item()
            rows.extend([i, j])
            cols.extend([j, i])
            weight = torch.rand(1, dtype=dtype).item() * 10 + 0.1
            vals.extend([weight, weight])

        # Add additional random edges
        edges_added = set()
        for i in range(1, num_nodes):
            for j in range(i):
                edges_added.add((j, i))

        attempts = 0
        max_attempts = num_nodes * num_nodes

        while len(edges_added) < num_edges and attempts < max_attempts:
            i = torch.randint(0, num_nodes, (1,)).item()
            j = torch.randint(0, num_nodes, (1,)).item()
            attempts += 1

            if i != j and (min(i, j), max(i, j)) not in edges_added:
                edges_added.add((min(i, j), max(i, j)))
                rows.extend([i, j])
                cols.extend([j, i])
                weight = torch.rand(1, dtype=dtype).item() * 10 + 0.1
                vals.extend([weight, weight])

        # Create sparse CSR tensor
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=dtype)

        # First create COO then convert to CSR
        self.input = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes)
        ).to_sparse_csr()

    def time_kruskal(self, num_nodes, density, dtype):
        beignet.kruskal(self.input)

    def peak_memory_kruskal(self, num_nodes, density, dtype):
        beignet.kruskal(self.input)
