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
    method using BFS to find augmenting paths. This implementation
    supports batched operations.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the capacity matrix with
        shape (num_nodes, num_nodes) for single graphs, or
        (batch_size, num_nodes, num_nodes) for batched graphs.
        Non-zero entries represent edge capacities.
    source : Tensor
        Source node indices. For single graphs, a scalar tensor.
        For batched operation, a 1D tensor with shape (batch_size,).
    sink : Tensor
        Sink node indices. For single graphs, a scalar tensor.
        For batched operation, a 1D tensor with shape (batch_size,).

    Returns
    -------
    max_flow : Tensor
        Maximum flow values. For single graphs, a scalar tensor.
        For batched operation, a 1D tensor with shape (batch_size,).
    """
    # Check if we have a batched operation
    if graph.dim() == 3:  # Batched CSR tensor
        batch_size = graph.shape[0]
        device = graph.device
        dtype = graph.dtype

        if source.dim() != 1 or source.numel() != batch_size:
            raise ValueError("Source tensor must be 1D with length matching batch size")
        if sink.dim() != 1 or sink.numel() != batch_size:
            raise ValueError("Sink tensor must be 1D with length matching batch size")

        # Pre-allocate result tensor
        result = torch.zeros(batch_size, dtype=dtype, device=device)

        # Process each graph in the batch
        for batch_idx in range(batch_size):
            current_graph = graph[batch_idx]
            current_source = source[batch_idx]
            current_sink = sink[batch_idx]
            max_flow = _single_graph_edmonds_karp(
                current_graph, current_source, current_sink
            )
            result[batch_idx] = max_flow

        return result
    else:
        # Single graph operation
        return _single_graph_edmonds_karp(graph, source, sink)


def _single_graph_edmonds_karp(graph: Tensor, source: Tensor, sink: Tensor) -> Tensor:
    """Internal function for single graph Edmonds-Karp algorithm."""
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
