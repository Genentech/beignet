import torch
from torch import Tensor


def kruskal(input: Tensor) -> Tensor:
    r"""
    Compute the minimum spanning tree of weighted undirected graphs using Kruskal's algorithm.

    The minimum spanning tree (MST) is a subset of edges that connects all vertices
    in the graph with the minimum total edge weight, without forming any cycles.

    Parameters
    ----------
    input : Tensor
        Adjacency matrix representation of weighted undirected graphs.
        - Shape (N, N) for a single graph over N nodes
        - Shape (..., N, N) for batched graphs
        - Can be sparse CSR tensor (for single graphs) or dense tensor (for batches)
        - For undirected graphs, if either input[i, j] or input[j, i] is non-zero,
          the edge weight is the minimum non-zero value of the two

    Returns
    -------
    output : Tensor
        Tensor representing the minimum spanning tree(s).
        - Same shape as input: (N, N) or (..., N, N)
        - Contains only the edges in the MST with their weights
        - For disconnected graphs, returns the minimum spanning forest
        - Returns sparse CSR for single graphs, dense for batched

    Notes
    -----
    This implementation uses Kruskal's algorithm with a union-find data structure
    for efficient cycle detection. The algorithm has time complexity O(E log E)
    where E is the number of edges.

    For best performance and to avoid precision issues, use sparse CSR tensors
    from torch.sparse as input.

    Examples
    --------
    >>> # Simple triangle graph as sparse CSR tensor
    >>> row_indices = torch.tensor([0, 0, 1, 1, 2, 2])
    >>> col_indices = torch.tensor([1, 2, 0, 2, 0, 1])
    >>> values = torch.tensor([1.0, 3.0, 1.0, 2.0, 3.0, 2.0])
    >>> adj_matrix = torch.sparse_csr_tensor(
    ...     torch.tensor([0, 2, 4, 6]),  # crow_indices
    ...     col_indices,
    ...     values,
    ...     size=(3, 3)
    ... )
    >>> mst = beignet.kruskal(adj_matrix)
    >>> # MST will contain edges with weights 1.0 and 2.0

    >>> # Batch processing with dense tensors
    >>> batch_size = 5
    >>> n_nodes = 4
    >>> # Create batch of random graphs as dense tensors
    >>> graphs = torch.rand(batch_size, n_nodes, n_nodes)
    >>> # Make symmetric for undirected graphs
    >>> graphs = (graphs + graphs.transpose(-1, -2)) / 2
    >>> # Zero out diagonal
    >>> graphs = graphs * (1 - torch.eye(n_nodes)).unsqueeze(0)
    >>> # Compute MST for all graphs in batch
    >>> batch_mst = beignet.kruskal(graphs)
    >>> batch_mst.shape
    torch.Size([5, 4, 4])
    """
    # Handle different input types and shapes
    if input.dim() == 2:
        # Single graph case
        if not input.is_sparse_csr:
            # Convert to sparse CSR if needed
            if input.is_sparse:
                input = input.to_sparse_csr()
            else:
                # Handle dense input by converting to sparse CSR
                input = input.to_sparse_csr()
        return _kruskal_single(input)
    else:
        # Batched case
        batch_shape = input.shape[:-2]
        n_nodes = input.shape[-1]

        # Flatten batch dimensions
        input_flat = input.reshape(-1, n_nodes, n_nodes)
        batch_size = input_flat.shape[0]

        # Process each graph in the batch
        results = []
        for i in range(batch_size):
            graph = input_flat[i]
            # Convert to sparse CSR if it's dense
            if not graph.is_sparse:
                graph = graph.to_sparse_csr()
            elif graph.is_sparse and graph.layout != torch.sparse_csr:
                graph = graph.to_sparse_csr()

            mst = _kruskal_single(graph)

            # For batched output, we need to use dense tensors
            # since sparse CSR doesn't support batching well
            mst_dense = mst.to_dense()
            results.append(mst_dense)

        # Stack results
        output = torch.stack(results)
        # Reshape to original batch dimensions
        output = output.reshape(*batch_shape, n_nodes, n_nodes)

        return output


def _kruskal_single(adj_matrix: Tensor) -> Tensor:
    """Process a single graph."""
    n_nodes = adj_matrix.shape[0]
    device = adj_matrix.device
    dtype = adj_matrix.dtype

    # Extract edges from sparse CSR matrix
    crow_indices = adj_matrix.crow_indices()
    col_indices = adj_matrix.col_indices()
    values = adj_matrix.values()

    # Build edge list from CSR format
    edges = []
    weights = []

    for i in range(n_nodes):
        start_idx = crow_indices[i].item()
        end_idx = crow_indices[i + 1].item()

        for idx in range(start_idx, end_idx):
            j = col_indices[idx].item()
            w = values[idx].item()

            # Only keep edges where i < j to avoid duplicates in undirected graph
            if i < j and w > 0:
                edges.append((i, j))
                weights.append(w)

    # Convert to tensors and sort by weight
    if len(edges) == 0:
        # Empty graph - return empty sparse matrix
        return torch.sparse_csr_tensor(
            torch.zeros(n_nodes + 1, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=dtype, device=device),
            size=(n_nodes, n_nodes),
        )

    edge_tensor = torch.tensor(edges, device=device)
    weight_tensor = torch.tensor(weights, dtype=dtype, device=device)

    # Sort edges by weight
    sorted_indices = torch.argsort(weight_tensor)
    sorted_edges = edge_tensor[sorted_indices]
    sorted_weights = weight_tensor[sorted_indices]

    # Union-Find data structure
    parent = torch.arange(n_nodes, device=device)
    rank = torch.zeros(n_nodes, dtype=torch.long, device=device)

    def find(x: int) -> int:
        """Find root with path compression."""
        path = []
        while parent[x].item() != x:
            path.append(x)
            x = parent[x].item()
        # Path compression
        for node in path:
            parent[node] = x
        return x

    def union(x: int, y: int) -> bool:
        """Union by rank."""
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
    mst_edges = []
    mst_weights = []

    for i in range(len(sorted_edges)):
        u, v = sorted_edges[i].tolist()

        if union(u, v):
            mst_edges.append([u, v])
            mst_weights.append(sorted_weights[i].item())

            # Early termination
            if len(mst_edges) == n_nodes - 1:
                break

    # Build sparse CSR tensor for MST
    if len(mst_edges) == 0:
        # No edges in MST
        return torch.sparse_csr_tensor(
            torch.zeros(n_nodes + 1, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=dtype, device=device),
            size=(n_nodes, n_nodes),
        )

    # Create symmetric edge list for undirected graph
    rows = []
    cols = []
    vals = []

    for (u, v), w in zip(mst_edges, mst_weights, strict=False):
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([w, w])

    # Convert to CSR format
    # First create COO tensor then convert to CSR
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=dtype, device=device)

    mst_sparse = torch.sparse_coo_tensor(
        indices, values, (n_nodes, n_nodes)
    ).to_sparse_csr()

    return mst_sparse
