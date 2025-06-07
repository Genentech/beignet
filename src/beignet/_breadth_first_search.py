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
    This implementation supports batched operations and is optimized for
    performance and gradient compatibility.

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
        Breadth-first traversal order. For single graphs, shape (num_reachable,).
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
        order_pos = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Initialize with sources for all batches
        batch_indices = torch.arange(batch_size, device=device)
        visited[batch_indices, source] = True
        result[batch_indices, 0] = source
        order_pos[:] = 1

        # Current level nodes for each batch (initially just sources)
        current_level = torch.full(
            (batch_size, num_nodes), -1, dtype=dtype, device=device
        )
        current_level[batch_indices, 0] = source
        level_sizes = torch.ones(batch_size, dtype=torch.long, device=device)

        # BFS level-by-level traversal
        while level_sizes.sum() > 0:
            next_level = torch.full(
                (batch_size, num_nodes), -1, dtype=dtype, device=device
            )
            next_level_sizes = torch.zeros(batch_size, dtype=torch.long, device=device)

            # Process all batches in parallel
            for batch_idx in range(batch_size):
                if level_sizes[batch_idx] == 0:
                    continue

                current_graph = graph[batch_idx]
                crow_indices = current_graph.crow_indices()
                col_indices = current_graph.col_indices()

                # Process current level nodes
                for level_idx in range(level_sizes[batch_idx]):
                    node = current_level[batch_idx, level_idx]

                    # Get neighbors
                    start_idx = crow_indices[node]
                    end_idx = crow_indices[node + 1]

                    for edge_idx in range(start_idx, end_idx):
                        neighbor = col_indices[edge_idx]

                        if not visited[batch_idx, neighbor]:
                            visited[batch_idx, neighbor] = True
                            # Add to result order
                            if order_pos[batch_idx] < num_nodes:
                                result[batch_idx, order_pos[batch_idx]] = neighbor
                                order_pos[batch_idx] += 1
                            # Add to next level
                            if next_level_sizes[batch_idx] < num_nodes:
                                next_level[batch_idx, next_level_sizes[batch_idx]] = (
                                    neighbor
                                )
                                next_level_sizes[batch_idx] += 1

            # Prepare for next iteration
            current_level = next_level
            level_sizes = next_level_sizes

        return result
    else:
        # Single graph operation
        return _single_graph_bfs(graph, source)


def _single_graph_bfs(graph: Tensor, source: Tensor) -> Tensor:
    """Internal function for single graph BFS."""
    num_nodes = graph.shape[-1]
    device = graph.device
    dtype = torch.long

    # Track visited nodes
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    order_list = []

    # Initialize with source
    current_level = torch.tensor([source.item()], dtype=dtype, device=device)
    visited[source] = True
    order_list.append(source.item())

    # Get graph structure
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()

    # BFS level-by-level traversal
    while len(current_level) > 0:
        next_level = []

        # Process all nodes in current level
        for node_idx in current_level:
            node = node_idx.item()

            # Get neighbors efficiently using slice indexing
            start_idx = crow_indices[node]
            end_idx = crow_indices[node + 1]

            if start_idx < end_idx:
                # Get all neighbors at once
                neighbors = col_indices[start_idx:end_idx]

                # Filter unvisited neighbors efficiently
                for neighbor in neighbors:
                    neighbor_idx = neighbor.item()
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        next_level.append(neighbor_idx)
                        order_list.append(neighbor_idx)

        # Prepare next level
        if next_level:
            current_level = torch.tensor(next_level, dtype=dtype, device=device)
        else:
            break

    # Convert result to tensor efficiently
    return torch.tensor(order_list, dtype=dtype, device=device)
