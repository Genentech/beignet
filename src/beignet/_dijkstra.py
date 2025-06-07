import torch
from torch import Tensor


def dijkstra(
    graph: Tensor,
    source: Tensor,
) -> Tensor:
    r"""
    Computes shortest paths from source nodes using Dijkstra's algorithm.

    Dijkstra's algorithm finds shortest paths from source vertices to all
    other vertices in a weighted graph with non-negative edge weights.
    It is more efficient than Bellman-Ford for graphs without negative edges.
    This implementation supports batched operations.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the weighted adjacency matrix with
        shape (num_nodes, num_nodes) for single graphs, or
        (batch_size, num_nodes, num_nodes) for batched graphs.
        Non-zero entries represent edge weights. All edge weights must be non-negative.
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

        # Track visited nodes for all batches
        visited = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)

        # Dijkstra's main loop
        for _ in range(num_nodes):
            # Find unvisited node with minimum distance for each batch
            mask_value = torch.tensor(float("inf"), dtype=dtype, device=device)
            masked_distances = torch.where(visited, mask_value, distances)
            min_dists, current_nodes = torch.min(masked_distances, dim=1)

            # Check if any batch has reachable nodes left
            any_reachable = (min_dists < float("inf")).any()
            if not any_reachable:
                break

            # Mark current nodes as visited
            visited[batch_indices, current_nodes] = True

            # Update distances for each batch
            for batch_idx in range(batch_size):
                if min_dists[batch_idx] == float("inf"):
                    continue  # No reachable nodes left in this batch

                current = current_nodes[batch_idx]
                current_graph = graph[batch_idx]
                crow_indices = current_graph.crow_indices()
                col_indices = current_graph.col_indices()
                values = current_graph.values()

                # Get edges from current node
                start_idx = crow_indices[current]
                end_idx = crow_indices[current + 1]

                # Update distances to neighbors
                for edge_idx in range(start_idx, end_idx):
                    neighbor = col_indices[edge_idx]
                    edge_weight = values[edge_idx]

                    if not visited[batch_idx, neighbor]:
                        new_distance = distances[batch_idx, current] + edge_weight
                        if new_distance < distances[batch_idx, neighbor]:
                            distances[batch_idx, neighbor] = new_distance

        return distances
    else:
        # Single graph operation
        return _single_graph_dijkstra(graph, source)


def _single_graph_dijkstra(graph: Tensor, source: Tensor) -> Tensor:
    """Internal function for single graph Dijkstra."""
    num_nodes = graph.shape[-1]
    device = graph.device
    dtype = graph.dtype

    # Initialize distances to infinity
    distances = torch.full((num_nodes,), float("inf"), dtype=dtype, device=device)

    # Set source distance to 0
    distances[source] = 0

    # Track visited nodes
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # Work with sparse representation
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()
    values = graph.values()

    # Dijkstra's main loop
    for _ in range(num_nodes):
        # Find unvisited node with minimum distance
        masked_distances = torch.where(
            visited, torch.tensor(float("inf"), dtype=dtype, device=device), distances
        )
        min_dist, current = torch.min(masked_distances, dim=0)

        # If minimum distance is infinity, no more reachable nodes
        if min_dist == float("inf"):
            break

        # Mark current node as visited
        visited[current] = True

        # Get edges from current node
        start_idx = crow_indices[current]
        end_idx = crow_indices[current + 1]

        # Update distances to neighbors
        for edge_idx in range(start_idx, end_idx):
            neighbor = col_indices[edge_idx]
            edge_weight = values[edge_idx]

            if not visited[neighbor]:
                new_distance = distances[current] + edge_weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance

    return distances
