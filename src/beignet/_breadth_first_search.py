import torch
from torch import Tensor


def breadth_first_search(
    graph: Tensor,
    source: Tensor,
) -> Tensor:
    r"""
    Performs breadth-first search traversal of a graph.

    Breadth-first search (BFS) explores the graph level by level,
    visiting all neighbors of a vertex before moving to the next level.

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
        Breadth-first traversal order with shape (num_reachable,).
        Contains node indices in the order they were visited.
    """
    num_nodes = graph.shape[-1]
    device = graph.device

    # Track visited nodes
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # Queue for BFS (using a list and managing indices manually)
    queue = torch.zeros(num_nodes, dtype=torch.long, device=device)
    queue_start = 0
    queue_end = 0

    # Result order
    order = torch.zeros(num_nodes, dtype=torch.long, device=device)
    order_idx = 0

    # Start BFS from source
    source_idx = source.item()
    queue[queue_end] = source_idx
    queue_end += 1
    visited[source_idx] = True
    order[order_idx] = source_idx
    order_idx += 1

    # Get graph structure
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()

    # BFS main loop
    while queue_start < queue_end:
        current = queue[queue_start].item()
        queue_start += 1

        # Get neighbors of current node
        start_idx = crow_indices[current]
        end_idx = crow_indices[current + 1]

        for edge_idx in range(start_idx, end_idx):
            neighbor = col_indices[edge_idx].item()

            if not visited[neighbor]:
                visited[neighbor] = True
                queue[queue_end] = neighbor
                queue_end += 1
                order[order_idx] = neighbor
                order_idx += 1

    # Return only the visited nodes
    return order[:order_idx]
