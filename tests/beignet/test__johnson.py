import beignet.hypothesis.strategies
import hypothesis.strategies
import numpy
import scipy.sparse
import scipy.sparse.csgraph
import torch


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_johnson(data):
    input = data.draw(
        beignet.hypothesis.strategies.csr_array_no_negative_cycles(),
    )

    input.indices = input.indices.astype(numpy.int32)
    input.indptr = input.indptr.astype(numpy.int32)

    expected = scipy.sparse.csgraph.johnson(
        input,
        directed=True,
        indices=None,
        return_predecessors=False,
        unweighted=False,
    )

    torch.testing.assert_close(
        expected,
        expected,
    )
