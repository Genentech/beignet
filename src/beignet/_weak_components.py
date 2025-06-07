import torch
from torch import Tensor


def weak_components(
    graph: Tensor,
) -> Tensor:
    r"""
    Finds weakly connected components in a directed graph.

    A weakly connected component is a maximal set of vertices such that
    for every pair of vertices there is an undirected path between them
    (ignoring edge directions). This implementation supports batched operations.

    Parameters
    ----------
    graph : Tensor
        Sparse CSR tensor representing the adjacency matrix with
        shape (num_nodes, num_nodes) for single graphs, or
        (batch_size, num_nodes, num_nodes) for batched graphs.
        Non-zero entries represent edges.

    Returns
    -------
    labels : Tensor
        Component labels for each node. For single graphs, shape (num_nodes,).
        For batched operation, shape (batch_size, num_nodes).
        Nodes in the same component have the same label.
    """
    # Check if we have a batched operation
    if graph.dim() == 3:  # Batched CSR tensor
        batch_size = graph.shape[0]
        num_nodes = graph.shape[-1]
        device = graph.device

        # Pre-allocate result tensor
        result = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=device)

        # Process each graph in the batch
        for batch_idx in range(batch_size):
            current_graph = graph[batch_idx]
            labels = _single_graph_weak_components(current_graph)
            result[batch_idx] = labels

        return result
    else:
        # Single graph operation
        return _single_graph_weak_components(graph)


def _single_graph_weak_components(graph: Tensor) -> Tensor:
    """Internal function for single graph weak components."""
    num_nodes = graph.shape[-1]
    device = graph.device

    # Union-Find data structure
    parent = torch.arange(num_nodes, dtype=torch.long, device=device)
    rank = torch.zeros(num_nodes, dtype=torch.long, device=device)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return

        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # Get graph structure
    crow_indices = graph.crow_indices()
    col_indices = graph.col_indices()

    # Union all connected nodes (treat as undirected)
    for i in range(num_nodes):
        start_idx = crow_indices[i]
        end_idx = crow_indices[i + 1]

        for edge_idx in range(start_idx, end_idx):
            j = col_indices[edge_idx].item()
            union(i, j)

    # Create component labels
    # First, find all unique root components
    roots = torch.unique(
        torch.tensor([find(i).item() for i in range(num_nodes)], device=device)
    )

    # Create mapping from root to component label
    root_to_label = {}
    for idx, root in enumerate(roots):
        root_to_label[root.item()] = idx

    # Assign labels to all nodes
    labels = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for i in range(num_nodes):
        root = find(i).item()
        labels[i] = root_to_label[root]

    return labels
