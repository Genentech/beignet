import pytest
from beignet.transforms import Transform


class TestTransform:
    def test___init__(self):
        assert Transform()

    def test__check_inputs(self):
        with pytest.raises(NotImplementedError) as _:
            Transform()._check_inputs([])

    def test__get_params(self):
        assert True

    def test__transform(self):
        with pytest.raises(NotImplementedError) as _:
            Transform()._transform(None, {})

    def test__transformables(self):
        assert True

    def test_extra_repr(self):
        assert True

    def test_forward(self):
        assert True
