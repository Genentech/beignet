import os.path

import pytest


@pytest.fixture
def data_path(request):
    directory = os.path.dirname(request.module.__file__)

    return os.path.join(directory, "../data")
