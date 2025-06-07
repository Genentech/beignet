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
def test_breadth_first_search_single(data):
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

    source = data.draw(
        hypothesis.strategies.integers(
            min_value=0,
            max_value=n - 1,
        ),
    )

    torch_graph = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(graph.indptr),
        col_indices=torch.from_numpy(graph.indices),
        values=torch.from_numpy(graph.data).float(),
        size=graph.shape,
    )

    torch_source = torch.tensor(source)

    expected, _ = scipy.sparse.csgraph.breadth_first_order(
        graph,
        i_start=source,
        directed=True,
    )

    actual = beignet.breadth_first_search(torch_graph, torch_source)

    # BFS order may vary, so just check that all expected nodes are visited
    # Filter out -1 padding values
    actual_filtered = actual[actual != -1]
    expected_set = set(expected.tolist())
    actual_set = set(actual_filtered.tolist())

    assert expected_set == actual_set


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_breadth_first_search_batched(data):
    # Test batched operation
    batch_size = data.draw(hypothesis.strategies.integers(min_value=2, max_value=4))
    n = data.draw(hypothesis.strategies.integers(min_value=2, max_value=5))

    # Create batched data
    all_crow_indices = []
    all_col_indices = []
    all_values = []
    sources = []
    expected_results = []

    # Collect data for all graphs in batch
    for _ in range(batch_size):
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

        # Create scipy sparse matrix for expected result
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

        source = data.draw(
            hypothesis.strategies.integers(
                min_value=0,
                max_value=n - 1,
            ),
        )

        expected, _ = scipy.sparse.csgraph.breadth_first_order(
            graph,
            i_start=source,
            directed=True,
        )

        # Store data for batched tensor creation
        all_crow_indices.append(graph.indptr.tolist())
        all_col_indices.append(graph.indices.tolist())
        all_values.append(graph.data.tolist())
        sources.append(source)
        expected_results.append(expected)

    # Pad to same number of edges for batched tensor creation
    max_edges = max(len(col_indices) for col_indices in all_col_indices)

    padded_crow_indices = []
    padded_col_indices = []
    padded_values = []

    for crow, col, vals in zip(
        all_crow_indices, all_col_indices, all_values, strict=True
    ):
        # Pad col_indices and values to max_edges
        padded_col = col + [0] * (max_edges - len(col))
        padded_val = vals + [0.0] * (max_edges - len(vals))

        padded_crow_indices.append(crow)
        padded_col_indices.append(padded_col)
        padded_values.append(padded_val)

    # Create batched CSR tensor
    batched_crow_indices = torch.tensor(padded_crow_indices)
    batched_col_indices = torch.tensor(padded_col_indices)
    batched_values = torch.tensor(padded_values, dtype=torch.float32)
    batched_sources = torch.tensor(sources)

    batched_graph = torch.sparse_csr_tensor(
        crow_indices=batched_crow_indices,
        col_indices=batched_col_indices,
        values=batched_values,
        size=(batch_size, n, n),
    )

    # Run batched BFS
    actual_batch = beignet.breadth_first_search(batched_graph, batched_sources)

    # Verify each result in the batch
    for batch_idx in range(batch_size):
        actual = actual_batch[batch_idx]
        expected = expected_results[batch_idx]

        # Filter out -1 padding values
        actual_filtered = actual[actual != -1]
        expected_set = set(expected.tolist())
        actual_set = set(actual_filtered.tolist())

        assert expected_set == actual_set
