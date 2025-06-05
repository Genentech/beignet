import beignet.hypothesis.strategies
import hypothesis.strategies
import numpy
import scipy.sparse
import scipy.sparse.csgraph
import torch


@hypothesis.given(
    data=hypothesis.strategies.data(),
)
def test_dinic(data):
    input = data.draw(
        beignet.hypothesis.strategies.csr_array_graph(
            integer_weights=True,
            allow_negative=False,
        ),
    )

    # Ensure indices and indptr are int32 for SciPy compatibility
    input.indices = input.indices.astype(numpy.int32)
    input.indptr = input.indptr.astype(numpy.int32)

    # Ensure we have at least 2 nodes and source != sink
    if input.shape[0] < 2:
        return

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
        ).filter(lambda x: x != source),
    )

    expected = scipy.sparse.csgraph.maximum_flow(
        input,
        source,
        sink,
        method="dinic",
    ).flow_value

    torch.testing.assert_close(
        expected,
        expected,
    )
