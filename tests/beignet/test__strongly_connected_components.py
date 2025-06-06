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
def test_strongly_connected_components(data):
    # Generate small graphs for faster testing
    n = data.draw(hypothesis.strategies.integers(min_value=2, max_value=5))

    # Create a simple graph with fewer edges
    density = data.draw(hypothesis.strategies.floats(min_value=0.1, max_value=0.5))
    num_edges = max(1, int(density * n * n))

    # Generate edges
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

    # Create scipy sparse matrix
    graph = scipy.sparse.csr_array(
        (
            numpy.array(weights, dtype=numpy.float32),
            (
                numpy.array(row_indices, dtype=numpy.int32),
                numpy.array(col_indices, dtype=numpy.int32),
            ),
        ),
        shape=(n, n),
    )

    # Ensure indices are int32 for scipy compatibility
    graph.indices = graph.indices.astype(numpy.int32)
    graph.indptr = graph.indptr.astype(numpy.int32)

    torch_graph = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(graph.indptr),
        col_indices=torch.from_numpy(graph.indices),
        values=torch.from_numpy(graph.data).float(),
        size=graph.shape,
    )

    _, expected = scipy.sparse.csgraph.connected_components(
        graph,
        directed=True,
        connection="strong",
        return_labels=True,
    )

    actual = beignet.strongly_connected_components(torch_graph)

    # Check that both have same number of components
    expected_components = len(set(expected))
    actual_components = len(set(actual.tolist()))

    assert expected_components == actual_components
