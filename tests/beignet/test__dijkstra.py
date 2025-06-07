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
def test_dijkstra_single(data):
    # Generate small graphs for faster testing
    n = data.draw(hypothesis.strategies.integers(min_value=2, max_value=5))

    # Create a simple graph with fewer edges
    density = data.draw(hypothesis.strategies.floats(min_value=0.1, max_value=0.5))
    num_edges = max(1, int(density * n * n))

    # Generate edges with non-negative weights (required for Dijkstra)
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
        values=torch.from_numpy(graph.data).to(torch.get_default_dtype()),
        size=graph.shape,
    )

    torch_source = torch.tensor(source)

    torch.testing.assert_close(
        beignet.dijkstra(torch_graph, torch_source),
        torch.from_numpy(
            scipy.sparse.csgraph.dijkstra(
                graph,
                directed=True,
                indices=source,
                return_predecessors=False,
                unweighted=False,
                limit=numpy.inf,
                min_only=False,
            ),
        ).to(torch.get_default_dtype()),
        rtol=1e-4,
        atol=1e-4,
    )

    if graph.nnz > 0:
        values = torch.tensor(
            graph.data.astype(numpy.float32),
            requires_grad=True,
        )

        distances = beignet.dijkstra(
            torch.sparse_csr_tensor(
                crow_indices=torch.from_numpy(graph.indptr),
                col_indices=torch.from_numpy(graph.indices),
                values=values,
                size=graph.shape,
            ),
            torch_source,
        )

        loss = torch.sum(distances)

        if loss.requires_grad:
            loss.backward()

            assert values.grad is not None

            assert torch.isfinite(values.grad).all()


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_dijkstra_batched(data):
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

        # Generate edges with non-negative weights (required for Dijkstra)
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

        expected = scipy.sparse.csgraph.dijkstra(
            graph,
            directed=True,
            indices=source,
            return_predecessors=False,
            unweighted=False,
            limit=numpy.inf,
            min_only=False,
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
    batched_values = torch.tensor(padded_values, dtype=torch.get_default_dtype())
    batched_sources = torch.tensor(sources)

    batched_graph = torch.sparse_csr_tensor(
        crow_indices=batched_crow_indices,
        col_indices=batched_col_indices,
        values=batched_values,
        size=(batch_size, n, n),
    )

    # Run batched Dijkstra
    actual_batch = beignet.dijkstra(batched_graph, batched_sources)

    # Verify each result in the batch
    for batch_idx in range(batch_size):
        actual = actual_batch[batch_idx]
        expected = expected_results[batch_idx]

        torch.testing.assert_close(
            actual,
            torch.from_numpy(expected).to(torch.get_default_dtype()),
            rtol=1e-4,
            atol=1e-4,
        )
