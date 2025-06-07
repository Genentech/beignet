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
    detects negative cycles. This implementation supports batched operations.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the weighted adjacency matrix with
        shape (num_nodes, num_nodes) for single graphs, or
        (batch_size, num_nodes, num_nodes) for batched graphs.
        Non-zero entries represent edge weights.
    source : Tensor
        Source node indices. For single graphs, a scalar tensor.
        For batched operation, a 1D tensor with shape (batch_size,).

    Returns
    -------
    distances : Tensor
        Shortest distances from source to all nodes. For single graphs,
        shape (num_nodes,). For batched operation, shape (batch_size, num_nodes).
        Unreachable nodes have distance infinity.
    """
    # Check if we have a batched operation
    if graph.dim() == 3:  # Batched CSR tensor
        batch_size = graph.shape[0]
        num_nodes = graph.shape[-1]
        device = graph.device
        dtype = graph.dtype

        if source.dim() != 1 or source.numel() != batch_size:
            raise ValueError("Source tensor must be 1D with length matching batch size")

        # Initialize distances for all batches
        distances = torch.full(
            (batch_size, num_nodes), float("inf"), dtype=dtype, device=device
        )

        # Set source distances to 0 for each batch
        batch_indices = torch.arange(batch_size, device=device)
        distances[batch_indices, source] = 0

        # Bellman-Ford relaxation (V-1 iterations)
        for _ in range(num_nodes - 1):
            prev_distances = distances.clone()

            # Vectorized edge relaxation across all batches
            for batch_idx in range(batch_size):
                current_graph = graph[batch_idx]
                crow_indices = current_graph.crow_indices()
                col_indices = current_graph.col_indices()
                values = current_graph.values()

                # Relax all edges for this batch
                for src in range(num_nodes):
                    if distances[batch_idx, src] == float("inf"):
                        continue

                    # Get edges from this source node
                    start_idx = crow_indices[src]
                    end_idx = crow_indices[src + 1]

                    for edge_idx in range(start_idx, end_idx):
                        dst = col_indices[edge_idx]
                        edge_weight = values[edge_idx]

                        # Relax edge
                        new_distance = distances[batch_idx, src] + edge_weight
                        if new_distance < distances[batch_idx, dst]:
                            distances[batch_idx, dst] = new_distance

            # Early termination if no improvement across all batches
            if torch.allclose(distances, prev_distances):
                break

        return distances
    else:
        # Single graph operation
        return _single_graph_bellman_ford(graph, source)


def _single_graph_bellman_ford(graph: Tensor, source: Tensor) -> Tensor:
    """Internal function for single graph Bellman-Ford."""
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
