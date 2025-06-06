import beignet
import beignet.hypothesis.strategies
import hypothesis.strategies
import numpy
import scipy.sparse
import scipy.sparse.csgraph
import torch


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_kruskal(data):
    # Generate small graphs for faster testing
    n = data.draw(hypothesis.strategies.integers(min_value=2, max_value=5))

    # Create a simple undirected graph by ensuring symmetric edges
    density = data.draw(hypothesis.strategies.floats(min_value=0.1, max_value=0.5))
    num_edges = max(1, int(density * n * n))

    # Generate edges with positive weights
    weights = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.floats(min_value=0.1, max_value=10.0),
            min_size=num_edges,
            max_size=num_edges,
        )
    )

    row_indices = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.integers(0, n - 1),
            min_size=num_edges,
            max_size=num_edges,
        )
    )

    col_indices = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.integers(0, n - 1),
            min_size=num_edges,
            max_size=num_edges,
        )
    )

    # Create symmetric edges for undirected graph
    all_weights = weights + weights
    all_rows = row_indices + col_indices
    all_cols = col_indices + row_indices

    # Create scipy sparse matrix
    graph = scipy.sparse.csr_array(
        (
            numpy.array(all_weights, dtype=numpy.float32),
            (
                numpy.array(all_rows, dtype=numpy.int32),
                numpy.array(all_cols, dtype=numpy.int32),
            ),
        ),
        shape=(n, n),
    )

    # Remove duplicates and self-loops
    graph.eliminate_zeros()
    graph.setdiag(0)

    # Ensure indices are int32 for scipy compatibility
    graph.indices = graph.indices.astype(numpy.int32)
    graph.indptr = graph.indptr.astype(numpy.int32)

    torch_graph = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(graph.indptr),
        col_indices=torch.from_numpy(graph.indices),
        values=torch.from_numpy(graph.data).float(),
        size=graph.shape,
    )

    expected = scipy.sparse.csgraph.minimum_spanning_tree(
        graph,
        overwrite=False,
    )

    actual = beignet.kruskal(torch_graph)

    # For MST, check that both have same total weight (divide by 2 for bidirectional)
    expected_weight = expected.data.sum()
    actual_weight = actual.values().sum() / 2  # Our MST is bidirectional

    torch.testing.assert_close(
        actual_weight,
        torch.tensor(expected_weight, dtype=actual_weight.dtype),
        rtol=1e-3,
        atol=1e-3,
    )
