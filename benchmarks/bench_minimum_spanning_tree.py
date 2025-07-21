import os

import torch

import beignet


class TimeMinimumSpanningTree:
    params = ([10, 50, 100], [0.1, 0.5, 1.0], [torch.float32, torch.float64])
    param_names = ["num_nodes", "density", "dtype"]

    def setup(self, num_nodes, density, dtype):
        torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42)))

        # Generate connected graph
        # First create a tree to ensure connectivity
        tree_edges = []
        for i in range(1, num_nodes):
            j = torch.randint(0, i, (1,)).item()
            tree_edges.append((j, i))

        # Add additional edges based on density
        num_possible_edges = num_nodes * (num_nodes - 1) // 2
        num_additional_edges = int((num_possible_edges - (num_nodes - 1)) * density)

        all_edges = set()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                all_edges.add((i, j))

        for edge in tree_edges:
            all_edges.discard(edge)
            all_edges.discard((edge[1], edge[0]))

        additional_edges = []
        if num_additional_edges > 0 and all_edges:
            additional_edges = list(all_edges)
            indices = torch.randperm(len(additional_edges))[
                : min(num_additional_edges, len(additional_edges))
            ]
            additional_edges = [additional_edges[idx] for idx in indices]

        selected_edges = tree_edges + additional_edges

        # Create edge list (bidirectional)
        edge_list = []
        for u, v in selected_edges:
            edge_list.extend([[u, v], [v, u]])

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).T
        self.edge_weight = torch.rand(self.edge_index.shape[1], dtype=dtype) * 10 + 0.1

    def time_minimum_spanning_tree(self, num_nodes, density, dtype):
        beignet.minimum_spanning_tree(num_nodes, self.edge_index, self.edge_weight)


class PeakMemoryMinimumSpanningTree:
    params = ([10, 50, 100], [0.1, 0.5, 1.0], [torch.float32, torch.float64])
    param_names = ["num_nodes", "density", "dtype"]

    def setup(self, num_nodes, density, dtype):
        torch.manual_seed(int(os.environ.get("BEIGNET_BENCHMARK_SEED", 42)))

        # Generate connected graph (same as above)
        tree_edges = []
        for i in range(1, num_nodes):
            j = torch.randint(0, i, (1,)).item()
            tree_edges.append((j, i))

        num_possible_edges = num_nodes * (num_nodes - 1) // 2
        num_additional_edges = int((num_possible_edges - (num_nodes - 1)) * density)

        all_edges = set()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                all_edges.add((i, j))

        for edge in tree_edges:
            all_edges.discard(edge)
            all_edges.discard((edge[1], edge[0]))

        additional_edges = []
        if num_additional_edges > 0 and all_edges:
            additional_edges = list(all_edges)
            indices = torch.randperm(len(additional_edges))[
                : min(num_additional_edges, len(additional_edges))
            ]
            additional_edges = [additional_edges[idx] for idx in indices]

        selected_edges = tree_edges + additional_edges

        edge_list = []
        for u, v in selected_edges:
            edge_list.extend([[u, v], [v, u]])

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).T
        self.edge_weight = torch.rand(self.edge_index.shape[1], dtype=dtype) * 10 + 0.1

    def peakmem_minimum_spanning_tree(self, num_nodes, density, dtype):
        beignet.minimum_spanning_tree(num_nodes, self.edge_index, self.edge_weight)
