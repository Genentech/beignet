import beignet.hypothesis.strategies
import hypothesis.strategies
import scipy.sparse
import scipy.sparse.csgraph
import torch


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_breadth_first_search(data):
    input = data.draw(
        beignet.hypothesis.strategies.csr_array(),
    )

    source = data.draw(
        hypothesis.strategies.integers(
            min_value=0,
            max_value=input.shape[0] - 1,
        ),
    )

    expected, _ = scipy.sparse.csgraph.breadth_first_order(
        input,
        i_start=source,
        directed=True,
    )

    torch.testing.assert_close(expected, expected)
