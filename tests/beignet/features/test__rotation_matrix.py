import pytest
import torch
from beignet.features import RotationMatrix
from torch import Size


class TestRotationMatrix:
    @pytest.fixture
    def rotation_matrix(self) -> RotationMatrix:
        return RotationMatrix(
            torch.tensor(
                [
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ],
            ),
        )

    def test___new__(self):
        result = RotationMatrix(
            torch.tensor(
                [
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ],
            ),
        )

        assert isinstance(result, RotationMatrix)

        assert result.shape == Size([1, 3, 3])

        with pytest.raises(ValueError):
            RotationMatrix(torch.tensor([1, 2, 3]))

    def test___repr__(self, rotation_matrix: RotationMatrix):
        assert isinstance(rotation_matrix.__repr__(), str)

    def test__wrap(self):
        result = RotationMatrix._wrap(
            torch.tensor(
                [
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ],
            ),
        )

        assert isinstance(result, RotationMatrix)

        assert result.shape == Size([1, 3, 3])

    def test_wrap_like(self, rotation_matrix: RotationMatrix):
        result = RotationMatrix.wrap_like(
            rotation_matrix,
            torch.tensor(
                [
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ],
            ),
        )

        assert isinstance(result, RotationMatrix)

        assert result.shape == Size([1, 3, 3])
