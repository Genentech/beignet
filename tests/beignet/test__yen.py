import beignet.hypothesis.strategies
import hypothesis.strategies
import scipy.sparse
import scipy.sparse.csgraph
import torch


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_yen(data):
    input = data.draw(
        beignet.hypothesis.strategies.csr_array_no_negative_cycles(),
    )

    source = data.draw(
        hypothesis.strategies.integers(
            min_value=0,
            max_value=input.shape[0] - 1,
        ),
    )

    sink = data.draw(
        hypothesis.strategies.integers(
            min_value=0,
            max_value=input.shape[0] - 1,
        ),
    )

    K = data.draw(
        hypothesis.strategies.integers(
            min_value=1,
            max_value=5,
        ),
    )

    expected = scipy.sparse.csgraph.yen(
        input,
        source,
        sink,
        K=K,
        directed=True,
        return_predecessors=False,
        unweighted=False,
    )

    torch.testing.assert_close(
        expected,
        expected,
    )
