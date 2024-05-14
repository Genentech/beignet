import os
from unittest import mock

import pytest
from omegaconf import DictConfig
from prescient.io import load_omega_config

TEST_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


@mock.patch("prescient.io._load_omega_config.S3FileSystem")
@mock.patch("prescient.io._load_omega_config.yaml.load", return_value={})
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="")
@pytest.mark.parametrize(
    "config_dir, config_name, compose, overrides",
    [
        pytest.param("s3://bucket/foo", "config.yaml", False, None, id="S3 path"),
        pytest.param(
            os.path.abspath(TEST_CONFIG_DIR),
            "test.yaml",
            True,
            ["+foo=bar"],
            id="Absolute path with config_name, compose=True",
        ),
        pytest.param(
            os.path.abspath(TEST_CONFIG_DIR),
            "test.yaml",
            False,
            None,
            id="Absolute path, compose=False",
        ),
        pytest.param(
            TEST_CONFIG_DIR,
            "test.yaml",
            True,
            ["+foo=bar"],
            id="Relative path, compose=True",
        ),
        pytest.param(
            TEST_CONFIG_DIR,
            "test.yaml",
            False,
            None,
            id="Relative path, compose=False",
        ),
        pytest.param(
            TEST_CONFIG_DIR,
            "test",
            False,
            None,
            id="Relative path with config_name without yaml suffix, compose=False",
        ),
    ],
)
def test_load_omega_config(
    mock_yaml_load,
    mock_s3_file_system,
    mock_builtin_open,
    config_dir,
    config_name,
    compose,
    overrides,
):
    mock_s3_file_system.open = mock.mock_open(read_data="")

    result = load_omega_config(
        config_dir=config_dir,
        config_name=config_name,
        compose=compose,
        overrides=overrides,
    )
    assert isinstance(result, DictConfig)


@pytest.mark.parametrize(
    "config_dir, config_name, compose, overrides",
    [
        pytest.param(
            TEST_CONFIG_DIR,
            None,
            False,
            ["+foo=bar"],
            id="compose=False with overrides",
        ),
        pytest.param(
            f"{TEST_CONFIG_DIR}/test.yaml",
            None,
            False,
            None,
            id="config dir is a file",
        ),
    ],
)
def test_load_omega_config_errors(
    config_dir,
    config_name,
    compose,
    overrides,
):
    with pytest.raises(ValueError):
        load_omega_config(
            config_dir=config_dir,
            config_name=config_name,
            compose=compose,
            overrides=overrides,
        )
