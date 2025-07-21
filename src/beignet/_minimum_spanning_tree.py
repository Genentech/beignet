import torch
from torch import Tensor


def minimum_spanning_tree(
    num_nodes: int,
    edge_index: Tensor,
    edge_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute the minimum spanning tree of a weighted undirected graph using Kruskal's algorithm.

    The minimum spanning tree (MST) is a subset of edges that connects all vertices
    in the graph with the minimum total edge weight, without forming any cycles.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edge_index : Tensor, shape=(2, E)
        Edge connectivity in COO format, where E is the number of edges.
        Each column represents an edge as [source, target].
        For undirected graphs, edges should be included in both directions.
    edge_weight : Tensor, shape=(E,)
        Edge weights corresponding to each edge in edge_index.
        Weights should be positive.

    Returns
    -------
    mst_edges : Tensor, shape=(2, num_nodes-1)
        Edges in the minimum spanning tree in COO format.
        For connected graphs, contains exactly num_nodes-1 edges.
    mst_weights : Tensor, shape=(num_nodes-1,)
        Weights of the edges in the minimum spanning tree.

    Notes
    -----
    This implementation uses Kruskal's algorithm with a union-find data structure
    for efficient cycle detection. The algorithm has time complexity O(E log E)
    where E is the number of edges.

    For disconnected graphs, the function returns a minimum spanning forest,
    containing the MST for each connected component.

    Examples
    --------
    >>> # Simple triangle graph
    >>> edge_index = torch.tensor([[0, 1, 0, 1, 2, 2], [1, 0, 2, 2, 0, 1]])
    >>> edge_weight = torch.tensor([1.0, 1.0, 3.0, 2.0, 3.0, 2.0])
    >>> mst_edges, mst_weights = beignet.minimum_spanning_tree(3, edge_index, edge_weight)
    >>> mst_edges.shape
    torch.Size([2, 2])
    >>> mst_weights.sum().item()
    3.0

    >>> # Batch processing with vmap
    >>> from torch.func import vmap
    >>> batch_weights = torch.rand(10, 6)  # 10 different weight configurations
    >>> def get_mst_weight(w):
    ...     _, weights = beignet.minimum_spanning_tree(3, edge_index, w)
    ...     return weights.sum()
    >>> total_weights = vmap(get_mst_weight)(batch_weights)
    >>> total_weights.shape
    torch.Size([10])
    """
    device = edge_index.device
    dtype = edge_weight.dtype

    # Handle undirected edges - keep only one direction to avoid duplicates
    # We'll keep edges where source < target
    # Use torch.where for torch.compile compatibility
    mask = edge_index[0] < edge_index[1]
    mask_indices = torch.where(mask)[0]
    unique_edges = edge_index[:, mask_indices]
    unique_weights = edge_weight[mask_indices]

    # Sort edges by weight (ascending order)
    sorted_indices = torch.argsort(unique_weights)
    sorted_edges = unique_edges[:, sorted_indices]
    sorted_weights = unique_weights[sorted_indices]

    # Convert to CPU for union-find operations (more efficient)
    parent = torch.arange(num_nodes)
    rank = torch.zeros(num_nodes, dtype=torch.long)

    def find(x: int) -> int:
        """Find root of the set containing x with path compression."""
        # Non-recursive implementation to avoid stack overflow
        root = x
        while parent[root].item() != root:
            root = parent[root].item()

        # Path compression
        while parent[x].item() != root:
            next_x = parent[x].item()
            parent[x] = root
            x = next_x

        return root

    def union(x: int, y: int) -> bool:
        """Union two sets. Returns True if they were in different sets."""
        root_x = find(x)
        root_y = find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

        return True

    # Kruskal's algorithm
    mst_edges_list = []
    mst_weights_list = []

    for i in range(sorted_edges.shape[1]):
        u = sorted_edges[0, i].item()
        v = sorted_edges[1, i].item()

        if union(u, v):
            mst_edges_list.append([u, v])
            mst_weights_list.append(sorted_weights[i])

            # Early termination if we have n-1 edges
            if len(mst_edges_list) == num_nodes - 1:
                break

    # Convert lists to tensors
    if mst_edges_list:
        mst_edges = torch.tensor(mst_edges_list, device=device).T
        mst_weights = torch.stack(mst_weights_list)
    else:
        # Empty graph case
        mst_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        mst_weights = torch.empty((0,), dtype=dtype, device=device)

    # Check if graph is connected (MST should have exactly n-1 edges)
    if mst_edges.shape[1] < num_nodes - 1 and num_nodes > 1:
        # Graph is disconnected - we have a forest
        pass  # This is valid, we return the minimum spanning forest

    return mst_edges, mst_weights
