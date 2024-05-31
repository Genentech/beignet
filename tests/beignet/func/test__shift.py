import hypothesis.strategies as st
import torch
from hypothesis import given

from beignet.func._molecular_dynamics._partition.__shift import _shift


@st.composite
def _shift_strategy(draw):
    max_dims = draw(st.integers(min_value=2, max_value=3))
    shape = draw(
        st.lists(st.integers(min_value=2, max_value=5), min_size=max_dims,
                 max_size=max_dims))

    tensor = draw(
        st.lists(
            st.floats(min_value=-5.0, max_value=5.0),
            min_size=torch.prod(torch.tensor(shape)).item(),
            max_size=torch.prod(torch.tensor(shape)).item()
        ).map(torch.tensor).map(lambda x: x.view(*shape))
    )

    shift_vec = draw(
        st.lists(
            st.integers(min_value=-shape[0], max_value=shape[0]),
            min_size=max_dims,
            max_size=max_dims
        ).map(torch.tensor)
    )

    return tensor, shift_vec


@given(_shift_strategy())
def test__shift(data):
    """
    Property-based test for the `_shift` function ensuring correct behavior for various inputs.
    """
    a, b = data

    result = _shift(a, b)

    assert result.shape == a.shape

    expected = torch.roll(a, shifts=tuple(b.numpy()), dims=tuple(range(len(b))))
    assert torch.equal(result, expected)
