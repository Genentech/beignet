import torch
from torch import Tensor


def depth_first_search(
    graph: Tensor,
    source: Tensor,
) -> Tensor:
    r"""
    Performs depth-first search traversal of a graph.

    Depth-first search (DFS) explores as far as possible along each branch
    before backtracking.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the adjacency matrix with
        shape (num_nodes, num_nodes). Non-zero entries represent edges.
    source : Tensor
        Source node index (scalar tensor).

    Returns
    -------
    order : Tensor
        Depth-first traversal order with shape (num_reachable,).
        Contains node indices in the order they were visited.
    """
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
