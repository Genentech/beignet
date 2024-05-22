import pytest
import torch
from beignet.features import Quaternion
from torch import Size


class TestQuaternion:
    @pytest.fixture
    def rotation_quaternion(self):
        return Quaternion(torch.tensor([0, 1, 0, 0]))

    def test___new__(self):
        result = Quaternion(torch.tensor([0, 1, 0, 0]))

        assert isinstance(result, Quaternion)

        assert result.shape == Size([1, 4])

        with pytest.raises(ValueError):
            Quaternion(torch.tensor([1, 2, 3]))

    def test___repr__(self, rotation_quaternion):
        assert isinstance(rotation_quaternion.__repr__(), str)

    def test__wrap(self):
        result = Quaternion._wrap(torch.tensor([0, 1, 0, 0]))

        assert isinstance(result, Quaternion)

        assert result.shape == Size([4])

    def test_wrap_like(self, rotation_quaternion):
        result = Quaternion.wrap_like(
            rotation_quaternion,
            torch.tensor([1, 0, 0, 0]),
        )

        assert isinstance(result, Quaternion)

        assert result.shape == Size([4])
