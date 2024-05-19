import copy

import pytest
import torch
from beignet.features import Feature


class TestFeature:
    def test___deepcopy__(self):
        feature = Feature(torch.tensor([1, 2, 3]))

        with pytest.raises(NotImplementedError):
            copy.deepcopy(feature)

    def test___torch_function__(self):
        feature = Feature(torch.tensor([1, 2, 3]))

        result = feature.__torch_function__(
            torch.add,
            (Feature, torch.Tensor),
            (feature, torch.tensor([1, 2, 3])),
        )

        assert not isinstance(result, Feature)

    def test__to_tensor(self):
        feature = Feature._to_tensor([1, 2, 3])

        assert torch.is_tensor(feature)

        assert not feature.requires_grad

    def test_device(self):
        feature = Feature(torch.tensor([1, 2, 3]))

        assert feature.device == feature.device

    def test_dtype(self):
        assert Feature(torch.tensor([1, 2, 3])).dtype == torch.int64

    def test_ndim(self):
        assert Feature(torch.tensor([1, 2, 3])).ndim == 1

    def test_shape(self):
        assert Feature(torch.tensor([1, 2, 3])).shape == (3,)

    def test_wrap_like(self):
        with pytest.raises(NotImplementedError):
            feature = Feature(torch.tensor([1, 2, 3]))

            Feature.wrap_like(feature, torch.tensor([1, 2, 3]))
