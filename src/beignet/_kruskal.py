import torch
from torch import Tensor


def kruskal(
    graph: Tensor,
) -> Tensor:
    r"""
    Computes minimum spanning tree using Kruskal's algorithm.

    Kruskal's algorithm finds a minimum spanning tree by greedily
    selecting the minimum weight edges that don't create cycles.
    This implementation supports batched operations.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the weighted adjacency matrix with
        shape (num_nodes, num_nodes) for single graphs, or
        (batch_size, num_nodes, num_nodes) for batched graphs.
        Non-zero entries represent edge weights.

    Returns
    -------
    mst : Tensor
        Sparse CSR tensor representing the minimum spanning tree. For single graphs,
        shape (num_nodes, num_nodes). For batched operation, shape
        (batch_size, num_nodes, num_nodes). Contains only the edges in the MST.
    """
    # Check if we have a batched operation
    if graph.dim() == 3:  # Batched CSR tensor
        batch_size = graph.shape[0]
        # num_nodes = graph.shape[-1]
        # device = graph.device

        # Process each graph in the batch
        batch_results = []
        for batch_idx in range(batch_size):
            current_graph = graph[batch_idx]
            mst = _single_graph_kruskal(current_graph)
            batch_results.append(mst)

        # Stack the results to create a batched sparse tensor
        return torch.stack(batch_results, dim=0)
    else:
        # Single graph operation
        return _single_graph_kruskal(graph)


def _single_graph_kruskal(graph: Tensor) -> Tensor:
    """Internal function for single graph Kruskal's algorithm."""
    num_nodes = graph.shape[-1]
    device = graph.device
    dtype = graph.dtype

    # Get graph structure
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()
    values = graph.values()

    # Create edge list with weights
    edges = []
    for i in range(num_nodes):
        start_idx = crow_indices[i]
        end_idx = crow_indices[i + 1]

        for edge_idx in range(start_idx, end_idx):
            j = col_indices[edge_idx].item()
            weight = values[edge_idx].item()

            # Only consider each edge once (undirected)
            if i <= j:
                edges.append((weight, i, j, edge_idx))

    # Sort edges by weight
    edges.sort(key=lambda x: x[0])

    # Union-Find data structure
    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    # Build MST
    mst_edges = []
    edges_added = 0

    for weight, i, j, _ in edges:
        if union(i, j):
            mst_edges.append((i, j, weight))
            edges_added += 1

            # MST has exactly n-1 edges
            if edges_added == num_nodes - 1:
                break

    # Build MST sparse tensor
    if not mst_edges:
        # Empty MST (no edges)
        mst_crow_indices = torch.zeros(
            num_nodes + 1, dtype=crow_indices.dtype, device=device
        )
        mst_col_indices = torch.zeros(0, dtype=col_indices.dtype, device=device)
        mst_values = torch.zeros(0, dtype=dtype, device=device)
    else:
        # Create MST edges (both directions for undirected graph)
        mst_row_indices = []
        mst_col_indices_list = []
        mst_values_list = []

        for i, j, weight in mst_edges:
            mst_row_indices.extend([i, j])
            mst_col_indices_list.extend([j, i])
            mst_values_list.extend([weight, weight])

        # Sort by row indices for CSR format
        edge_data = list(
            zip(mst_row_indices, mst_col_indices_list, mst_values_list, strict=False)
        )
        edge_data.sort(key=lambda x: (x[0], x[1]))

        # Build CSR arrays
        mst_crow_indices = torch.zeros(
            num_nodes + 1, dtype=crow_indices.dtype, device=device
        )
        current_row = 0
        for idx, (row, _, _) in enumerate(edge_data):
            while current_row <= row:
                mst_crow_indices[current_row] = idx
                current_row += 1
        while current_row <= num_nodes:
            mst_crow_indices[current_row] = len(edge_data)
            current_row += 1

        mst_col_indices = torch.tensor(
            [x[1] for x in edge_data], dtype=col_indices.dtype, device=device
        )
        mst_values = torch.tensor([x[2] for x in edge_data], dtype=dtype, device=device)

    return torch.sparse_csr_tensor(
        crow_indices=mst_crow_indices,
        col_indices=mst_col_indices,
        values=mst_values,
        size=graph.shape,
    )
