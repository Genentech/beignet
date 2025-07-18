import torch

from beignet.features import EulerAngle

from .._set_seed import set_seed


class BenchEulerAngle:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def setup(self, batch_size, dtype):
        set_seed()

        self.data = torch.randn(batch_size, 3, dtype=dtype)

        self.feature = EulerAngle(self.data)

    def time___init__(self, batch_size, dtype):
        EulerAngle(self.data)

    def peak_memory___init__(self, batch_size, dtype):
        EulerAngle(self.data)

    def time_shape(self, batch_size, dtype):
        _ = self.feature.shape

    def peak_memory_shape(self, batch_size, dtype):
        _ = self.feature.shape

    def time_device(self, batch_size, dtype):
        _ = self.feature.device

    def peak_memory_device(self, batch_size, dtype):
        _ = self.feature.device

    def time_dtype_property(self, batch_size, dtype):
        _ = self.feature.dtype

    def peak_memory_dtype_property(self, batch_size, dtype):
        _ = self.feature.dtype

    def time_ndim(self, batch_size, dtype):
        _ = self.feature.ndim

    def peak_memory_ndim(self, batch_size, dtype):
        _ = self.feature.ndim

    def time_wrap_like(self, batch_size, dtype):
        EulerAngle.wrap_like(self.feature, self.data)

    def peak_memory_wrap_like(self, batch_size, dtype):
        EulerAngle.wrap_like(self.feature, self.data)

    def time___repr__(self, batch_size, dtype):
        repr(self.feature)

    def peak_memory___repr__(self, batch_size, dtype):
        repr(self.feature)
