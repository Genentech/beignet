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

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the weighted adjacency matrix with
        shape (num_nodes, num_nodes). Non-zero entries represent edge weights.
        All edge weights must be non-negative.
    source : Tensor
        Source node index (scalar tensor).

    Returns
    -------
    distances : Tensor
        Shortest distances from source to all nodes with shape (num_nodes,).
        Unreachable nodes have distance infinity.
    """
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
