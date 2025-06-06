import torch
from torch import Tensor


def floyd_warshall(
    graph: Tensor,
) -> Tensor:
    r"""
    Computes all-pairs shortest paths using the Floyd-Warshall algorithm.

    The Floyd-Warshall algorithm finds shortest paths between all pairs of
    vertices in a weighted graph. It can handle negative edge weights but
    not negative cycles.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the weighted adjacency matrix with
        shape (num_nodes, num_nodes). Non-zero entries represent edge weights.

    Returns
    -------
    distances : Tensor
        Matrix of shortest distances between all pairs of nodes with shape
        (num_nodes, num_nodes). distances[i, j] is the shortest distance
        from node i to node j. Unreachable pairs have distance infinity.
    """
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
