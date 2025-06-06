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
def test_dinic(data):
    # Generate small graphs for faster testing
    n = data.draw(hypothesis.strategies.integers(min_value=2, max_value=4))

    # Ensure we have at least 2 nodes
    if n < 2:
        return

    # Create a simple graph with fewer edges and integer weights
    density = data.draw(hypothesis.strategies.floats(min_value=0.1, max_value=0.5))
    num_edges = max(1, int(density * n * n))

    # Generate edges with integer weights (required for flow)
    weights = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.integers(min_value=1, max_value=10),
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
            numpy.array(weights, dtype=numpy.int32),
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

    source = data.draw(
        hypothesis.strategies.integers(
            min_value=0,
            max_value=n - 1,
        ),
    )

    sink = data.draw(
        hypothesis.strategies.integers(
            min_value=0,
            max_value=n - 1,
        ).filter(lambda x: x != source),
    )

    torch_graph = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(graph.indptr),
        col_indices=torch.from_numpy(graph.indices),
        values=torch.from_numpy(graph.data).float(),
        size=graph.shape,
    )

    torch_source = torch.tensor(source)
    torch_sink = torch.tensor(sink)

    expected = scipy.sparse.csgraph.maximum_flow(
        graph,
        source,
        sink,
        method="dinic",
    ).flow_value

    actual = beignet.dinic(torch_graph, torch_source, torch_sink)

    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=actual.dtype),
        rtol=1e-4,
        atol=1e-4,
    )
