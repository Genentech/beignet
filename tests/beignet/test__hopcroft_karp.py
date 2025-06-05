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
    input = data.draw(
        beignet.hypothesis.strategies.csr_array(),
    )

    input.indices = input.indices.astype(numpy.int32)
    input.indptr = input.indptr.astype(numpy.int32)

    expected = scipy.sparse.csgraph.maximum_bipartite_matching(
        input,
        perm_type="row",
    )

    torch.testing.assert_close(
        expected,
        expected,
    )
