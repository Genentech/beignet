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
def test_hopcroft_karp(data):
    # Generate small bipartite graphs for faster testing
    n = data.draw(hypothesis.strategies.integers(min_value=4, max_value=6))

    # For bipartite graphs, ensure even number of nodes
    if n % 2 == 1:
        n += 1

    # Create a simple bipartite graph with edges only between partitions
    left_size = n // 2
    right_size = n - left_size

    density = data.draw(hypothesis.strategies.floats(min_value=0.1, max_value=0.5))
    num_edges = max(1, int(density * left_size * right_size))

    # Generate edges only between left and right partitions
    weights = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.floats(min_value=0.1, max_value=10.0),
            min_size=num_edges,
            max_size=num_edges,
        )
    )

    row_indices = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.integers(0, left_size - 1),  # Left partition
            min_size=num_edges,
            max_size=num_edges,
        )
    )

    col_indices = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.integers(left_size, n - 1),  # Right partition
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

    expected = scipy.sparse.csgraph.maximum_bipartite_matching(
        graph,
        perm_type="row",
    )

    actual = beignet.hopcroft_karp(torch_graph)

    # Compare matching sizes - count non-negative entries in scipy result
    expected_size = numpy.sum(expected >= 0)
    actual_size = actual.item()

    torch.testing.assert_close(
        torch.tensor(actual_size, dtype=torch.float32),
        torch.tensor(expected_size, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-4,
    )
