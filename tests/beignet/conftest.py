import os.path

import pytest
import torch

from beignet import default_dtype_manager


@pytest.fixture
def data_path(request):
    directory = os.path.dirname(request.module.__file__)

    return os.path.join(directory, "../data")


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(torch.float64, id="float64"),
        pytest.param(torch.float32, id="float32"),
    ],
)
def default_dtypes(request):
    with default_dtype_manager(request.param):
        yield


@pytest.fixture()
def float32(request):
    with default_dtype_manager(torch.float32):
        yield


@pytest.fixture()
def float64(request):
    with default_dtype_manager(torch.float64):
        yield
