import pytest

from beignet.func._interact import _merge_dictionaries


@pytest.fixture
def dictionary_1():
    return dict(
        {
            "foo": [1, 2],
            "bar": [3, 4],
        }
    )


@pytest.fixture
def dictionary_2():
    return dict(
        {
            "bar": [5, 6],
            "baz": [7, 8],
        }
    )


def test_merge_dictionaries(dictionary_1, dictionary_2):
    expected = dict(
        {
            "foo": [1, 2],
            "bar": [5, 6],
            "baz": [7, 8],
        }
    )

    assert _merge_dictionaries(dictionary_1, dictionary_2) == expected


def test_merge_dictionaries_ignore_unused_parameters(dictionary_1, dictionary_2):
    expected = dict(
        {
            "foo": [1, 2],
            "bar": [5, 6],
        }
    )

    assert (
        _merge_dictionaries(
            dictionary_1,
            dictionary_2,
            True,
        )
        == expected
    )
