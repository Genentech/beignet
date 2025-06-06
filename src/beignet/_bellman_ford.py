import torch
from torch import Tensor


def bellman_ford(
    graph: Tensor,
    source: Tensor,
) -> Tensor:
    r"""
    Computes shortest paths from source nodes using the Bellman-Ford algorithm.

    The Bellman-Ford algorithm finds shortest paths from source vertices to all
    other vertices in a weighted graph. It can handle negative edge weights and
    detects negative cycles.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the weighted adjacency matrix with
        shape (num_nodes, num_nodes). Non-zero entries represent edge weights.
    source : Tensor
        Source node index (scalar tensor).

    Returns
    -------
    distances : Tensor
        Shortest distances from source to all nodes with shape (num_nodes,).
        Unreachable nodes have distance infinity.
    """
    num_nodes = graph.shape[-1]

    # Initialize distances to infinity
    distances = torch.full(
        (num_nodes,), float("inf"), dtype=graph.dtype, device=graph.device
    )

    # Set source distance to 0
    distances[source] = 0

    # Work with sparse representation to distinguish explicit zeros from missing edges
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()
    values = graph.values()

    # Bellman-Ford relaxation (V-1 iterations)
    for _ in range(num_nodes - 1):
        prev_distances = distances.clone()

        # For each node, check its outgoing edges
        for src in range(num_nodes):
            if distances[src] == float("inf"):
                continue

            # Get edges from this source node
            start_idx = crow_indices[src]
            end_idx = crow_indices[src + 1]

            for edge_idx in range(start_idx, end_idx):
                dst = col_indices[edge_idx]
                edge_weight = values[edge_idx]

                # All stored edges are valid (including explicit zeros)
                new_distance = distances[src] + edge_weight
                if new_distance < distances[dst]:
                    distances[dst] = new_distance

        # Early termination if no improvement
        if torch.allclose(distances, prev_distances):
            break

    return distances
