import beignet.hypothesis.strategies
import hypothesis.strategies
import numpy
import scipy.sparse
import scipy.sparse.csgraph
import torch


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_dijkstra(data):
    input = data.draw(
        beignet.hypothesis.strategies.csr_array_no_negative_cycles(),
    )

    input.indices = input.indices.astype(numpy.int32)
    input.indptr = input.indptr.astype(numpy.int32)

    indices = data.draw(
        hypothesis.strategies.integers(
            min_value=0,
            max_value=input.shape[0] - 1,
        ),
    )

    expected = scipy.sparse.csgraph.dijkstra(
        input,
        directed=True,
        indices=indices,
        return_predecessors=False,
        unweighted=False,
        limit=numpy.inf,
        min_only=False,
    )

    torch.testing.assert_close(
        expected,
        expected,
    )
