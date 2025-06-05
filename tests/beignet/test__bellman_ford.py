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
    input = data.draw(
        beignet.hypothesis.strategies.csr_array_graph(
            dtypes=[
                numpy.int32,
                numpy.int64,
            ],
            allow_negative=False,
        ),
    )

    input.indices = input.indices.astype(numpy.int32)
    input.indptr = input.indptr.astype(numpy.int32)

    indices = data.draw(
        hypothesis.strategies.integers(
            min_value=0,
            max_value=input.shape[0] - 1,
        ),
    )

    expected = scipy.sparse.csgraph.bellman_ford(
        input,
        directed=True,
        indices=indices,
        return_predecessors=False,
        unweighted=False,
    )

    torch.testing.assert_close(
        expected,
        expected,
    )
