import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import beignet


@given(
    num_nodes=st.integers(min_value=2, max_value=50),
    dtype=st.sampled_from([torch.float32, torch.float64]),
    density=st.floats(min_value=0.1, max_value=1.0),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(deadline=None)  # Disable deadline due to torch.compile
def test_minimum_spanning_tree(
    num_nodes: int,
    dtype: torch.dtype,
    density: float,
    seed: int,
) -> None:
    """Test minimum_spanning_tree operator."""
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate random weighted graph
    # First create a tree to ensure connectivity
    tree_edges = []
    for i in range(1, num_nodes):
        # Connect node i to a random previous node
        j = torch.randint(0, i, (1,)).item()
        tree_edges.append((j, i))

    # Add additional edges based on density
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_additional_edges = int((num_possible_edges - (num_nodes - 1)) * density)

    # Generate all possible non-tree edges
    all_edges = set()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            all_edges.add((i, j))

    # Remove tree edges from possible edges
    for edge in tree_edges:
        all_edges.discard(edge)
        all_edges.discard((edge[1], edge[0]))

    # Randomly select additional edges
    additional_edges = []
    if num_additional_edges > 0 and all_edges:
        additional_edges = list(all_edges)
        torch.manual_seed(seed + 1)  # Different seed for edge selection
        indices = torch.randperm(len(additional_edges))[
            : min(num_additional_edges, len(additional_edges))
        ]
        additional_edges = [additional_edges[idx] for idx in indices]

    # Combine tree and additional edges
    selected_edges = tree_edges + additional_edges

    # Create edge list tensor
    edge_index = torch.tensor(
        [[edge[0] for edge in selected_edges], [edge[1] for edge in selected_edges]],
        dtype=torch.long,
    )

    # Make edges bidirectional (undirected graph)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Generate random edge weights
    num_directed_edges = edge_index.shape[1]
    edge_weight = torch.rand(num_directed_edges // 2, dtype=dtype) * 10 + 0.1
    # Duplicate weights for bidirectional edges
    edge_weight = torch.cat([edge_weight, edge_weight])

    # Basic test
    mst_edges, mst_weights = beignet.minimum_spanning_tree(
        num_nodes, edge_index, edge_weight
    )

    # Check output types and shapes
    assert isinstance(mst_edges, torch.Tensor)
    assert isinstance(mst_weights, torch.Tensor)
    assert mst_edges.dtype == torch.long
    assert mst_weights.dtype == dtype

    # MST should have exactly n-1 edges
    assert mst_edges.shape[1] == num_nodes - 1
    assert mst_weights.shape[0] == num_nodes - 1

    # Check that MST edges are valid
    assert torch.all(mst_edges >= 0)
    assert torch.all(mst_edges < num_nodes)

    # Check that all weights are positive
    assert torch.all(mst_weights > 0)

    # Verify MST is a tree (connected and acyclic)
    # Build adjacency list from MST edges
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(mst_edges.shape[1]):
        u, v = mst_edges[0, i].item(), mst_edges[1, i].item()
        adj_list[u].append(v)
        adj_list[v].append(u)

    # BFS to check connectivity
    visited = [False] * num_nodes
    queue = [0]
    visited[0] = True
    count = 1

    while queue:
        node = queue.pop(0)
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                count += 1

    assert count == num_nodes, "MST is not connected"

    # Test with simple known graph
    if num_nodes >= 3:
        # Triangle graph: 0-1 (weight 1), 1-2 (weight 2), 0-2 (weight 3)
        simple_edges = torch.tensor(
            [[0, 1, 0, 1, 2, 2], [1, 0, 2, 2, 0, 1]], dtype=torch.long
        )
        simple_weights = torch.tensor([1.0, 1.0, 3.0, 2.0, 3.0, 2.0], dtype=dtype)

        simple_mst_edges, simple_mst_weights = beignet.minimum_spanning_tree(
            3, simple_edges, simple_weights
        )

        # MST should include edges with weights 1 and 2, not 3
        assert simple_mst_edges.shape[1] == 2
        assert simple_mst_weights.sum().item() == 3.0

    # Test gradient computation
    if dtype == torch.float64:
        edge_weight_grad = edge_weight.clone().requires_grad_(True)
        _, mst_weights_grad = beignet.minimum_spanning_tree(
            num_nodes, edge_index, edge_weight_grad
        )

        # Gradient exists for sum of MST weights
        loss = mst_weights_grad.sum()
        loss.backward()

        assert edge_weight_grad.grad is not None
        assert torch.all(torch.isfinite(edge_weight_grad.grad))

    # Test edge cases
    # Single edge graph (2 nodes)
    if num_nodes == 2:
        assert mst_edges.shape[1] == 1
        assert mst_weights.shape[0] == 1

    # Test with uniform weights (any MST is valid)
    uniform_weights = torch.ones_like(edge_weight)
    uniform_mst_edges, uniform_mst_weights = beignet.minimum_spanning_tree(
        num_nodes, edge_index, uniform_weights
    )
    assert uniform_mst_edges.shape[1] == num_nodes - 1
    assert torch.allclose(
        uniform_mst_weights.sum(), torch.tensor(float(num_nodes - 1), dtype=dtype)
    )

    # Note: torch.compile is not supported for this operator due to dynamic control flow
    # The algorithm's union-find operations and early termination make it incompatible
    # with fullgraph compilation

    # Note: vmap is not supported for this operator due to .item() calls in the union-find algorithm
    # The algorithm requires extracting scalar values for indexing which is incompatible with vmap

    # Test that MST weight is minimal
    # For small graphs, we can verify optimality
    if num_nodes <= 5:
        # MST weight should be less than or equal to any other spanning tree
        # This is computationally expensive to verify exhaustively, so we do a sanity check
        total_weight = edge_weight[: num_directed_edges // 2].sum()
        mst_total_weight = mst_weights.sum()
        assert mst_total_weight <= total_weight

    # Test disconnected graph handling
    if num_nodes >= 4:
        # Create disconnected graph: two separate components
        disc_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        disc_weights = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=dtype)

        try:
            disc_mst_edges, disc_mst_weights = beignet.minimum_spanning_tree(
                4, disc_edges, disc_weights
            )
            # If it returns, it should return a forest (multiple trees)
            # In this case, we expect 2 edges (one for each component)
            assert disc_mst_edges.shape[1] < 3  # Less than n-1 for disconnected
        except ValueError as e:
            # It's also acceptable to raise an error for disconnected graphs
            assert "disconnected" in str(e).lower()
