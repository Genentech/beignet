import torch
from torch import Tensor


def hopcroft_karp(
    graph: Tensor,
) -> Tensor:
    r"""
    Computes maximum bipartite matching using Hopcroft-Karp algorithm.

    The Hopcroft-Karp algorithm finds a maximum matching in a bipartite graph.
    This is a simplified implementation that assumes the graph represents
    a bipartite graph where the first half of nodes are in one partition
    and the second half are in the other partition. This implementation
    supports batched operations.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the bipartite graph with
        shape (num_nodes, num_nodes) for single graphs, or
        (batch_size, num_nodes, num_nodes) for batched graphs.
        Non-zero entries represent edges. Assumes nodes 0 to num_nodes//2-1
        are in the left partition and nodes num_nodes//2 to num_nodes-1 are
        in the right partition.

    Returns
    -------
    matching_size : Tensor
        Size of the maximum matching. For single graphs, a scalar tensor.
        For batched operation, a 1D tensor with shape (batch_size,).
    """
    # Check if we have a batched operation
    if graph.dim() == 3:  # Batched CSR tensor
        batch_size = graph.shape[0]
        device = graph.device
        dtype = graph.dtype

        # Pre-allocate result tensor
        result = torch.zeros(batch_size, dtype=dtype, device=device)

        # Process each graph in the batch
        for batch_idx in range(batch_size):
            current_graph = graph[batch_idx]
            matching_size = _single_graph_hopcroft_karp(current_graph)
            result[batch_idx] = matching_size

        return result
    else:
        # Single graph operation
        return _single_graph_hopcroft_karp(graph)


def _single_graph_hopcroft_karp(graph: Tensor) -> Tensor:
    """Internal function for single graph Hopcroft-Karp algorithm."""
    num_nodes = graph.shape[-1]
    device = graph.device
    dtype = graph.dtype

    # For simplicity, assume bipartite graph is split in half
    left_size = num_nodes // 2
    right_size = num_nodes - left_size

    if left_size == 0 or right_size == 0:
        return torch.tensor(0, dtype=dtype, device=device)

    # Matching arrays: -1 means unmatched
    match_left = torch.full((left_size,), -1, dtype=torch.long, device=device)
    match_right = torch.full((right_size,), -1, dtype=torch.long, device=device)

    # Get graph structure
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()

    def bfs():
        """BFS to build level graph."""
        level = torch.full((left_size,), -1, dtype=torch.long, device=device)
        queue = torch.zeros(left_size, dtype=torch.long, device=device)
        queue_start = 0
        queue_end = 0

        # Add all unmatched left nodes to queue
        for u in range(left_size):
            if match_left[u] == -1:
                level[u] = 0
                queue[queue_end] = u
                queue_end += 1

        found_augmenting_path = False

        while queue_start < queue_end:
            u = queue[queue_start].item()
            queue_start += 1

            # Look at neighbors of u (only in right partition)
            start_idx = crow_indices[u]
            end_idx = crow_indices[u + 1]

            for edge_idx in range(start_idx, end_idx):
                v = col_indices[edge_idx].item()

                # Only consider edges to right partition
                if v >= left_size:
                    v_right = v - left_size

                    if match_right[v_right] == -1:
                        # Found augmenting path
                        found_augmenting_path = True
                    else:
                        matched_u = match_right[v_right]
                        if level[matched_u] == -1:
                            level[matched_u] = level[u] + 1
                            queue[queue_end] = matched_u
                            queue_end += 1

        return found_augmenting_path, level

    def dfs(u, level):
        """DFS to find augmenting paths."""
        if u == -1:
            return True

        start_idx = crow_indices[u]
        end_idx = crow_indices[u + 1]

        for edge_idx in range(start_idx, end_idx):
            v = col_indices[edge_idx].item()

            # Only consider edges to right partition
            if v >= left_size:
                v_right = v - left_size
                matched_u = match_right[v_right]

                if matched_u == -1 or (
                    level[matched_u] == level[u] + 1 and dfs(matched_u, level)
                ):
                    match_left[u] = v_right
                    match_right[v_right] = u
                    return True

        level[u] = -1
        return False

    # Hopcroft-Karp main loop
    matching = 0
    while True:
        found_path, level = bfs()
        if not found_path:
            break

        # Find augmenting paths using DFS
        for u in range(left_size):
            if match_left[u] == -1 and dfs(u, level):
                matching += 1

    return torch.tensor(matching, dtype=dtype, device=device)
