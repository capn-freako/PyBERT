"""Unit test coverage to make sure that the pybert can correctly save and load files."""

import logging
import pickle

import numpy as np
import yaml

from pybert import __version__
from pybert.pybert import PyBERT


def test_save_config_as_yaml(tmp_path):
    """Make sure that pybert can correctly generate a yaml file that can get reloaded."""
    app = PyBERT(run_simulation=False, gui=False)
    save_file = tmp_path.joinpath("config.yaml")
    app.save_configuration(save_file)

    assert save_file.exists()  # File was created.

    with open(save_file, "r", encoding="UTF-8") as saved_config_file:
        user_config = yaml.load(saved_config_file, Loader=yaml.Loader)
        assert user_config.version == __version__


def test_save_config_as_pickle(tmp_path):
    """Make sure that pybert can correctly generate a pickle file that can get reloaded."""
    app = PyBERT(run_simulation=False, gui=False)
    save_file = tmp_path.joinpath("config.pybert_cfg")
    app.save_configuration(save_file)

    assert save_file.exists()  # File was created.

    with open(save_file, "rb") as saved_config_file:
        user_config = pickle.load(saved_config_file)
        assert user_config.version == __version__


def test_save_config_as_invalid(tmp_path, caplog):
    """When given an unsupported file suffix, no file should be generated and an message logged."""
    caplog.set_level(logging.DEBUG)

    app = PyBERT(run_simulation=False, gui=False)
    save_file = tmp_path.joinpath("config.json")
    app.save_configuration(save_file)

    assert not save_file.exists()  # File should not have been created.
    assert "Pybert does not support this file type." in caplog.text


def test_save_results_as_pickle(tmp_path):
    """Make sure that pybert can correctly generate a waveform pickle file that can get reloaded."""
    app = PyBERT(run_simulation=False, gui=False)
    save_file = tmp_path.joinpath("results.pybert_data")
    app.save_results(save_file)

    assert save_file.exists()  # File was created.

    with open(save_file, "rb") as saved_results_file:
        results = pickle.load(saved_results_file)
        assert results.the_data.arrays


def test_load_config_from_yaml(tmp_path):
    """Make sure that pybert can correctly load a yaml file."""
    app = PyBERT(run_simulation=False, gui=False)
    save_file = tmp_path.joinpath("config.yaml")
    app.save_configuration(save_file)
    TEST_NUMBER_OF_BITS = 1234

    # Modify the saved yaml file.
    with open(save_file, "r", encoding="UTF-8") as saved_config_file:
        user_config = yaml.load(saved_config_file, Loader=yaml.Loader)
        user_config.nbits = TEST_NUMBER_OF_BITS  # Normally, 8000
    with open(save_file, "w", encoding="UTF-8") as saved_config_file:
        yaml.dump(user_config, saved_config_file)

    app.load_configuration(save_file)
    assert app.nbits == TEST_NUMBER_OF_BITS


def test_load_config_from_pickle(tmp_path):
    """Make sure that pybert can correctly load a pickle file."""
    app = PyBERT(run_simulation=False, gui=False)
    save_file = tmp_path.joinpath("config.pybert_cfg")
    app.save_configuration(save_file)
    TEST_PATTERN_LENGTH = 31

    # Modify the saved pickle file.
    with open(save_file, "rb") as saved_config_file:
        user_config = pickle.load(saved_config_file)
        user_config.pattern_len = TEST_PATTERN_LENGTH  # Normally, 127
    with open(save_file, "wb") as saved_config_file:
        pickle.dump(user_config, saved_config_file)

    app.load_configuration(save_file)
    assert app.pattern_len == TEST_PATTERN_LENGTH


def test_load_config_from_invalid(tmp_path, caplog):
    """When given an unsupported file suffix, no file should be read and an message logged."""
    caplog.set_level(logging.DEBUG)

    app = PyBERT(run_simulation=False, gui=False)
    save_file = tmp_path.joinpath("config.json")
    app.load_configuration(save_file)

    assert "Pybert does not support this file type." in caplog.text


def test_load_results_from_pickle(tmp_path, caplog):
    """Make sure that pybert can correctly load a pickle file."""
    caplog.set_level(logging.DEBUG)
    app = PyBERT(run_simulation=True, gui=False)
    save_file = tmp_path.joinpath("config.pybert_data")
    app.save_results(save_file)

    # Modify the saved pickle file.
    with open(save_file, "rb") as saved_results_file:
        user_results = pickle.load(saved_results_file)
        user_results.the_data.update_data({"chnl_h": np.array([1, 2, 3, 4])})
    with open(save_file, "wb") as saved_results_file:
        pickle.dump(user_results, saved_results_file)

    caplog.clear()
    app.load_results(save_file)
    # pybert doesn't directly reload the waveform back into the same plot.
    # instead if creates a reference plot to compare old vs. new.
    assert app.plotdata.get_data("chnl_h_ref").size == 4
