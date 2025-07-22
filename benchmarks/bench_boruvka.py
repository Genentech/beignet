import os

import torch

import beignet


class BenchBoruvka:
    """Benchmark for Borůvka's minimum spanning tree algorithm."""

    params = ([10, 50, 100], [0.1, 0.5, 1.0], [torch.float32, torch.float64])
    param_names = ["num_nodes", "density", "dtype"]

    def setup(self, num_nodes, density, dtype):
        torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 0x00B1)))

        # Create a random connected graph as dense tensor
        # (sparse CSR batching is not supported)
        n_edges = int(num_nodes * (num_nodes - 1) * density / 2)
        n_edges = max(num_nodes - 1, n_edges)  # Ensure connected

        # Start with a tree to ensure connectivity
        self.adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=dtype)

        # Create spanning tree
        for i in range(1, num_nodes):
            j = torch.randint(0, i, (1,)).item()
            weight = torch.rand(1, dtype=dtype).item() * 10 + 0.1
            self.adj_matrix[i, j] = weight
            self.adj_matrix[j, i] = weight

        # Add random edges
        edges_added = num_nodes - 1
        attempts = 0
        max_attempts = n_edges * 10

        while edges_added < n_edges and attempts < max_attempts:
            i = torch.randint(0, num_nodes, (1,)).item()
            j = torch.randint(0, num_nodes, (1,)).item()
            attempts += 1

            if i != j and self.adj_matrix[i, j] == 0:
                weight = torch.rand(1, dtype=dtype).item() * 10 + 0.1
                self.adj_matrix[i, j] = weight
                self.adj_matrix[j, i] = weight
                edges_added += 1

        # Compile the function (fullgraph=False due to dynamic control flow)
        self.compiled_fn = torch.compile(beignet.boruvka, fullgraph=False)

    def time_boruvka(self, num_nodes, density, dtype):
        """Time Borůvka's algorithm execution."""
        beignet.boruvka(self.adj_matrix)

    def time_boruvka_compiled(self, num_nodes, density, dtype):
        """Time compiled Borůvka's algorithm execution."""
        self.compiled_fn(self.adj_matrix)

    def peak_memory_boruvka(self, num_nodes, density, dtype):
        """Measure peak memory usage of Borůvka's algorithm."""
        beignet.boruvka(self.adj_matrix)
