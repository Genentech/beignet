import torch
from torch import Tensor


def floyd_warshall(
    graph: Tensor,
) -> Tensor:
    r"""
    Computes all-pairs shortest paths using the Floyd-Warshall algorithm.

    The Floyd-Warshall algorithm finds shortest paths between all pairs of
    vertices in a weighted graph. It can handle negative edge weights but
    not negative cycles. This implementation supports efficient batched operations.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the weighted adjacency matrix with
        shape (num_nodes, num_nodes) for single graphs, or
        (batch_size, num_nodes, num_nodes) for batched graphs.
        Non-zero entries represent edge weights.

    Returns
    -------
    distances : Tensor
        Matrix of shortest distances between all pairs of nodes. For single graphs,
        shape (num_nodes, num_nodes). For batched operation,
        shape (batch_size, num_nodes, num_nodes). distances[..., i, j] is the
        shortest distance from node i to node j.
        Unreachable pairs have distance infinity.
    """
    # Check if we have a batched operation
    if graph.dim() == 3:  # Batched CSR tensor
        batch_size = graph.shape[0]
        num_nodes = graph.shape[-1]
        device = graph.device
        dtype = graph.dtype

        # Initialize distance matrices with infinity
        distances = torch.full(
            (batch_size, num_nodes, num_nodes), float("inf"), dtype=dtype, device=device
        )

        # Set diagonal to zero (distance from node to itself)
        distances.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Fully vectorized extraction of sparse graph data
        # Convert sparse CSR tensors to dense for efficient batch processing
        for batch_idx in range(batch_size):
            current_graph = graph[batch_idx]
            # Convert to dense and copy non-diagonal elements
            dense_graph = current_graph.to_dense()
            # Only copy non-diagonal elements
            mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
            distances[batch_idx][mask] = dense_graph[mask]

        # Vectorized Floyd-Warshall main loop across all batches
        for k in range(num_nodes):
            # Compute new distances through intermediate node k for all batches at once
            # distances[:, i, k] + distances[:, k, j] for all i, j
            dist_ik = distances[:, :, k : k + 1]  # Shape: (batch_size, num_nodes, 1)
            dist_kj = distances[:, k : k + 1, :]  # Shape: (batch_size, 1, num_nodes)
            new_distances = dist_ik + dist_kj  # Broadcasting

            # Update distances where new path is shorter
            distances = torch.minimum(distances, new_distances)

        return distances
    else:
        # Single graph operation
        return _single_graph_floyd_warshall(graph)


def _single_graph_floyd_warshall(graph: Tensor) -> Tensor:
    """Internal function for single graph Floyd-Warshall."""
    num_nodes = graph.shape[-1]
    device = graph.device
    dtype = graph.dtype

    # Initialize distance matrix with infinity
    distances = torch.full(
        (num_nodes, num_nodes), float("inf"), dtype=dtype, device=device
    )

    # Set diagonal to zero (distance from node to itself)
    distances.fill_diagonal_(0)

    # Fill in direct edge weights
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()
    values = graph.values()

    for i in range(num_nodes):
        start_idx = crow_indices[i]
        end_idx = crow_indices[i + 1]

        for edge_idx in range(start_idx, end_idx):
            j = col_indices[edge_idx]
            weight = values[edge_idx]
            # Don't overwrite diagonal elements (self-loops should be 0)
            if i != j:
                distances[i, j] = weight

    # Floyd-Warshall main loop
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Check if path through k is shorter
                new_distance = distances[i, k] + distances[k, j]
                if new_distance < distances[i, j]:
                    distances[i, j] = new_distance

    return distances
