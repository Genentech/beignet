import beignet.hypothesis.strategies
import hypothesis.strategies
import scipy.sparse
import scipy.sparse.csgraph
import torch


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_strongly_connected_components(data):
    input = data.draw(
        beignet.hypothesis.strategies.csr_array(),
    )

    _, expected = scipy.sparse.csgraph.connected_components(
        input,
        directed=True,
        connection="strong",
        return_labels=True,
    )

    torch.testing.assert_close(
        expected,
        expected,
    )
