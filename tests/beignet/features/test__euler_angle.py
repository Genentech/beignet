import pytest
import torch
from beignet.features import EulerAngle
from torch import Size


class TestEulerAngle:
    @pytest.fixture
    def euler_angles(self) -> EulerAngle:
        return EulerAngle(torch.tensor([[1, 2, 3]]))

    def test___new__(self):
        result = EulerAngle(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, EulerAngle)

        assert result.shape == Size([1, 3])

        with pytest.raises(ValueError):
            EulerAngle(torch.tensor(5))

        with pytest.raises(ValueError):
            EulerAngle(torch.tensor([1, 2]))

    def test___repr__(self, euler_angles: EulerAngle):
        assert isinstance(euler_angles.__repr__(), str)

    def test__wrap(self):
        result = EulerAngle._wrap(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, EulerAngle)

        assert result.shape == Size([1, 3])

    def test_wrap_like(self, euler_angles: EulerAngle):
        result = EulerAngle.wrap_like(
            euler_angles,
            torch.tensor([[1, 2, 3]]),
        )

        assert isinstance(result, EulerAngle)

        assert result.shape == Size([1, 3])
