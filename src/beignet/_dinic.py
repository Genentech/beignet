import torch
from torch import Tensor


def dinic(
    graph: Tensor,
    source: Tensor,
    sink: Tensor,
) -> Tensor:
    r"""
    Computes maximum flow using Dinic's algorithm.

    Dinic's algorithm uses level graphs and blocking flows to find
    maximum flow more efficiently than Edmonds-Karp.
    This implementation supports batched operations.

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
            max_flow = _single_graph_dinic(current_graph, current_source, current_sink)
            result[batch_idx] = max_flow

        return result
    else:
        # Single graph operation
        return _single_graph_dinic(graph, source, sink)


def _single_graph_dinic(graph: Tensor, source: Tensor, sink: Tensor) -> Tensor:
    """Internal function for single graph Dinic's algorithm."""
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

    def bfs_level_graph():
        """Build level graph using BFS."""
        level = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        level[source_idx] = 0

        queue = torch.zeros(num_nodes, dtype=torch.long, device=device)
        queue_start = 0
        queue_end = 0

        queue[queue_end] = source_idx
        queue_end += 1

        while queue_start < queue_end:
            current = queue[queue_start].item()
            queue_start += 1

            for neighbor in range(num_nodes):
                if level[neighbor] == -1 and residual[current, neighbor] > 0:
                    level[neighbor] = level[current] + 1
                    queue[queue_end] = neighbor
                    queue_end += 1

        return level[sink_idx] != -1, level

    def dfs_blocking_flow(node, sink, level, min_flow):
        """Find blocking flow using DFS."""
        if node == sink:
            return min_flow

        flow = 0.0
        for neighbor in range(num_nodes):
            if level[neighbor] == level[node] + 1 and residual[node, neighbor] > 0:
                pushed = dfs_blocking_flow(
                    neighbor,
                    sink,
                    level,
                    min(min_flow - flow, residual[node, neighbor].item()),
                )

                if pushed > 0:
                    residual[node, neighbor] -= pushed
                    residual[neighbor, node] += pushed
                    flow += pushed

                    if flow == min_flow:
                        break

        return flow

    # Dinic's main loop
    while True:
        # Build level graph
        reachable, level = bfs_level_graph()
        if not reachable:
            break

        # Find blocking flows
        while True:
            flow = dfs_blocking_flow(source_idx, sink_idx, level, float("inf"))
            if flow == 0:
                break
            max_flow_value += flow

    return torch.tensor(max_flow_value, dtype=dtype, device=device)
