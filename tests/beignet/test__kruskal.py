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
    input = data.draw(
        beignet.hypothesis.strategies.csr_array(),
    )

    input.indices = input.indices.astype(numpy.int32)
    input.indptr = input.indptr.astype(numpy.int32)

    expected = scipy.sparse.csgraph.minimum_spanning_tree(
        input,
        overwrite=False,
    )

    torch.testing.assert_close(
        expected.data,
        expected.data,
    )
