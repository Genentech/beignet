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
def test_bellman_ford(data):
    n = data.draw(
        hypothesis.strategies.integers(
            min_value=2,
            max_value=5,
        ),
    )

    density = data.draw(
        hypothesis.strategies.floats(
            min_value=0.1,
            max_value=0.5,
        ),
    )
    num_edges = max(1, int(density * n * n))

    weights = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.floats(
                min_value=0.1,
                max_value=10.0,
            ),
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

    torch_source = torch.tensor(
        source,
    )

    torch.testing.assert_close(
        beignet.bellman_ford(
            torch_graph,
            torch_source,
        ),
        torch.from_numpy(
            scipy.sparse.csgraph.bellman_ford(
                graph,
                directed=True,
                indices=source,
                return_predecessors=False,
                unweighted=False,
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

        distances = beignet.bellman_ford(
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
