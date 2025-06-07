import torch
from torch import Tensor


def depth_first_search(
    graph: Tensor,
    source: Tensor,
) -> Tensor:
    r"""
    Performs depth-first search traversal of a graph.

    Depth-first search (DFS) explores as far as possible along each branch
    before backtracking. This implementation supports batched operations.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the adjacency matrix with
        shape (num_nodes, num_nodes) for single graphs, or
        (batch_size, num_nodes, num_nodes) for batched graphs.
        Non-zero entries represent edges.
    source : Tensor
        Source node indices. For single graphs, a scalar tensor.
        For batched operation, a 1D tensor with shape (batch_size,).

    Returns
    -------
    order : Tensor
        Depth-first traversal order. For single graphs, shape (num_reachable,).
        For batched operation, shape (batch_size, max_reachable).
        Unreachable positions in batched results are filled with -1.
    """
    # Check if we have a batched operation
    if graph.dim() == 3:  # Batched CSR tensor
        batch_size = graph.shape[0]
        num_nodes = graph.shape[-1]
        device = graph.device
        dtype = torch.long

        if source.dim() != 1 or source.numel() != batch_size:
            raise ValueError("Source tensor must be 1D with length matching batch size")

        # Pre-allocate result tensor
        result = torch.full((batch_size, num_nodes), -1, dtype=dtype, device=device)

        # Track visited nodes for all batches
        visited = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)

        # Stacks for DFS (using tensors)
        stacks = torch.full((batch_size, num_nodes), -1, dtype=dtype, device=device)
        stack_sizes = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Result order tracking
        order_pos = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Initialize stacks with source nodes
        batch_indices = torch.arange(batch_size, device=device)
        stacks[batch_indices, 0] = source
        stack_sizes[:] = 1

        # DFS main loop
        while stack_sizes.sum() > 0:
            # Process all batches in parallel
            for batch_idx in range(batch_size):
                if stack_sizes[batch_idx] == 0:
                    continue

                # Pop from stack
                stack_sizes[batch_idx] -= 1
                current = stacks[batch_idx, stack_sizes[batch_idx]]

                if not visited[batch_idx, current]:
                    visited[batch_idx, current] = True
                    # Add to result order
                    if order_pos[batch_idx] < num_nodes:
                        result[batch_idx, order_pos[batch_idx]] = current
                        order_pos[batch_idx] += 1

                    # Add neighbors to stack in reverse order
                    current_graph = graph[batch_idx]
                    crow_indices = current_graph.crow_indices()
                    col_indices = current_graph.col_indices()

                    start_idx = crow_indices[current]
                    end_idx = crow_indices[current + 1]

                    # Add neighbors in reverse order so they're processed in
                    # forward order
                    for edge_idx in range(end_idx - 1, start_idx - 1, -1):
                        neighbor = col_indices[edge_idx]

                        if not visited[batch_idx, neighbor]:
                            if stack_sizes[batch_idx] < num_nodes:
                                stacks[batch_idx, stack_sizes[batch_idx]] = neighbor
                                stack_sizes[batch_idx] += 1

        return result
    else:
        # Single graph operation
        return _single_graph_dfs(graph, source)


def _single_graph_dfs(graph: Tensor, source: Tensor) -> Tensor:
    """Internal function for single graph DFS."""
    num_nodes = graph.shape[-1]
    device = graph.device

    # Track visited nodes
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # Stack for DFS (using a list and managing indices manually)
    stack = torch.zeros(num_nodes, dtype=torch.long, device=device)
    stack_size = 0

    # Result order
    order = torch.zeros(num_nodes, dtype=torch.long, device=device)
    order_idx = 0

    # Start DFS from source
    source_idx = source.item()
    stack[stack_size] = source_idx
    stack_size += 1

    # Get graph structure
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()

    # DFS main loop
    while stack_size > 0:
        # Pop from stack
        stack_size -= 1
        current = stack[stack_size].item()

        if not visited[current]:
            visited[current] = True
            order[order_idx] = current
            order_idx += 1

            # Add neighbors to stack (in reverse order for consistent traversal)
            start_idx = crow_indices[current]
            end_idx = crow_indices[current + 1]

            # Add neighbors in reverse order so they're processed in forward order
            for edge_idx in range(end_idx - 1, start_idx - 1, -1):
                neighbor = col_indices[edge_idx].item()

                if not visited[neighbor]:
                    stack[stack_size] = neighbor
                    stack_size += 1

    # Return only the visited nodes
    return order[:order_idx]
