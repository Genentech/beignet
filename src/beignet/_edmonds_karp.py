import torch
from torch import Tensor


def edmonds_karp(
    graph: Tensor,
    source: Tensor,
    sink: Tensor,
) -> Tensor:
    r"""
    Computes maximum flow using the Edmonds-Karp algorithm.

    The Edmonds-Karp algorithm is an implementation of the Ford-Fulkerson
    method using BFS to find augmenting paths.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the capacity matrix with
        shape (num_nodes, num_nodes). Non-zero entries represent edge capacities.
    source : Tensor
        Source node index (scalar tensor).
    sink : Tensor
        Sink node index (scalar tensor).

    Returns
    -------
    max_flow : Tensor
        Maximum flow value (scalar tensor).
    """
    num_nodes = graph.shape[-1]
    device = graph.device
    dtype = graph.dtype

    source_idx = source.item()
    sink_idx = sink.item()

    # Convert sparse graph to dense residual capacity matrix for simplicity
    residual = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)

    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()
    values = graph.values()

    # Fill residual capacity matrix
    for i in range(num_nodes):
        start_idx = crow_indices[i]
        end_idx = crow_indices[i + 1]

        for edge_idx in range(start_idx, end_idx):
            j = col_indices[edge_idx].item()
            capacity = values[edge_idx]
            residual[i, j] = capacity

    max_flow_value = 0.0

    # Edmonds-Karp main loop
    while True:
        # BFS to find augmenting path
        parent = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        queue = torch.zeros(num_nodes, dtype=torch.long, device=device)
        queue_start = 0
        queue_end = 0

        # Start BFS from source
        queue[queue_end] = source_idx
        queue_end += 1
        visited[source_idx] = True

        # BFS to find path to sink
        found_path = False
        while queue_start < queue_end and not found_path:
            current = queue[queue_start].item()
            queue_start += 1

            for neighbor in range(num_nodes):
                if not visited[neighbor] and residual[current, neighbor] > 0:
                    visited[neighbor] = True
                    parent[neighbor] = current
                    queue[queue_end] = neighbor
                    queue_end += 1

                    if neighbor == sink_idx:
                        found_path = True
                        break

        # If no path found, we're done
        if not found_path:
            break

        # Find minimum capacity along the path
        path_flow = float("inf")
        current = sink_idx
        while current != source_idx:
            prev = parent[current].item()
            path_flow = min(path_flow, residual[prev, current].item())
            current = prev

        # Update residual capacities
        current = sink_idx
        while current != source_idx:
            prev = parent[current].item()
            residual[prev, current] -= path_flow
            residual[current, prev] += path_flow
            current = prev

        max_flow_value += path_flow

    return torch.tensor(max_flow_value, dtype=dtype, device=device)
