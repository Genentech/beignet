import copy

import pytest
import torch
from beignet.features import Feature


class TestFeature:
    def test__to_tensor(self, feature: Feature):
        result = Feature._to_tensor([1, 2, 3])

        assert torch.is_tensor(result)

        assert not result.requires_grad

    def test_wrap_like(self):
        with pytest.raises(NotImplementedError):
            Feature.wrap_like(None, None)

    def test___torch_function__(self, feature: Feature):
        result = feature.__torch_function__(
            torch.add,
            (Feature, torch.Tensor),
            (feature, torch.tensor([1, 2, 3])),
        )

        assert not isinstance(result, Feature)

    def test_device(self, feature: Feature):
        assert feature.device == feature.device

    def test_ndim(self, feature: Feature):
        assert feature.ndim == 1

    def test_dtype(self, feature: Feature):
        assert feature.dtype == torch.int64

    def test_shape(self, feature: Feature):
        assert feature.shape == (3,)

    def test___deepcopy__(self, feature: Feature):
        with pytest.raises(NotImplementedError):
            copy.deepcopy(feature)
