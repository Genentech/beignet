import pytest
import torch
from beignet.features import RotationVector
from torch import Size


class TestRotationVector:
    @pytest.fixture
    def rotation_vector(self) -> RotationVector:
        return RotationVector(torch.tensor([[1, 2, 3]]))

    def test___new__(self):
        result = RotationVector(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, RotationVector)

        assert result.shape == Size([1, 3])

        with pytest.raises(ValueError):
            RotationVector(torch.tensor(5))

        with pytest.raises(ValueError):
            RotationVector(torch.tensor([1, 2]))

    def test___repr__(self, rotation_vector):
        assert isinstance(rotation_vector.__repr__(), str)

    def test__wrap(self):
        result = RotationVector._wrap(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, RotationVector)

        assert result.shape == Size([1, 3])

    def test_wrap_like(self, rotation_vector):
        result = RotationVector.wrap_like(
            rotation_vector,
            torch.tensor([[1, 2, 3]]),
        )

        assert isinstance(result, RotationVector)

        assert result.shape == Size([1, 3])
