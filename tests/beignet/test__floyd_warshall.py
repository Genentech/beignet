import beignet.hypothesis.strategies
import hypothesis.strategies
import scipy.sparse
import scipy.sparse.csgraph
import torch


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_floyd_warshall(data):
    input = data.draw(
        beignet.hypothesis.strategies.csr_array_no_negative_cycles(),
    )

    expected = scipy.sparse.csgraph.floyd_warshall(
        input,
        directed=True,
        return_predecessors=False,
        unweighted=False,
        overwrite=False,
    )

    torch.testing.assert_close(
        expected,
        expected,
    )
