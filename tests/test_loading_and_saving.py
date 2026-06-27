"""Unit test coverage to make sure that the pybert can correctly save and load files."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from pybert import __version__
from pybert.configuration import InvalidFileType, PyBertCfg
from pybert.pybert import PyBERT


def test_load_old_bool_traits_compat(tmp_path: Path):
    """Old configs using Bool traits (use_ch_file, rx_use_ibis, tx_use_ibis) load correctly
    via the backward-compat mapping added in configuration.py:222-229."""
    dut = PyBERT(run_simulation=False, gui=False)

    # Build a minimal old-format config via pickle, injecting the old Bool keys.
    config = PyBertCfg(dut, "string_time", "test.test.test")
    config.use_ch_file  = True   # old name; should map to inter_sel = "single"
    config.rx_use_ibis  = True   # old name; should map to rx_sel    = "ibis"
    config.tx_use_ibis  = False  # old name; should map to tx_sel    = "native"
    # Remove the new-style attrs so load_from_file only sees the old keys.
    for attr in ("inter_sel", "rx_sel", "tx_sel"):
        config.__dict__.pop(attr, None)

    save_file = tmp_path / "compat.pybert_cfg"
    with open(save_file, "wb") as f:
        pickle.dump(config, f)

    dut2 = PyBERT(run_simulation=False, gui=False)
    dut2.load_configuration(save_file)

    assert dut2.inter_sel == "single", f"Expected inter_sel='single', got {dut2.inter_sel!r}"
    assert dut2.rx_sel    == "ibis",   f"Expected rx_sel='ibis',   got {dut2.rx_sel!r}"
    assert dut2.tx_sel    == "native", f"Expected tx_sel='native', got {dut2.tx_sel!r}"


@pytest.mark.parametrize("filepath_converter", [str, Path])
@pytest.mark.usefixtures("dut")
def test_save_config_as_yaml(dut, filepath_converter, tmp_path: Path):
    """Make sure that pybert can correctly generate a yaml file that can get reloaded."""
    save_file = tmp_path.joinpath("config.yaml")
    dut.save_configuration(filepath_converter(save_file))

    assert save_file.exists()  # File was created.

    with open(save_file, "r", encoding="UTF-8") as saved_config_file:
        user_config = yaml.load(saved_config_file, Loader=yaml.Loader)
        assert user_config.version == __version__


@pytest.mark.usefixtures("dut")
def test_save_config_as_invalid(dut, tmp_path: Path):
    """When given an unsupported file suffix, no file should be generated and an message logged."""
    save_file = tmp_path.joinpath("config.json")
    dut.save_configuration(save_file)

    assert not save_file.exists()  # File should not have been created.
    assert "This filetype is not currently supported." in dut.console_log


@pytest.mark.usefixtures("dut")
def test_save_results_as_pickle(dut, tmp_path: Path):
    """Make sure that pybert can correctly generate a waveform pickle file that can get reloaded."""
    save_file = tmp_path.joinpath("results.pybert_data")
    dut.save_results(save_file)

    assert save_file.exists()  # File was created.

    with open(save_file, "rb") as saved_results_file:
        results = pickle.load(saved_results_file)
        assert results.the_data.arrays


@pytest.mark.parametrize("filepath_converter", [str, Path])
@pytest.mark.usefixtures("dut")
def test_load_config_from_yaml(dut, filepath_converter, tmp_path: Path):
    """Make sure that pybert can correctly load a yaml file."""
    save_file = tmp_path.joinpath("config.yaml")
    dut.save_configuration(save_file)

    # Modify the saved yaml file.
    with open(save_file, "r", encoding="UTF-8") as saved_config_file:
        user_config = yaml.load(saved_config_file, Loader=yaml.Loader)
        # Change a lot of settings throughout the different tabs of the application.
        user_config.eye_bits = 1234  # Normally 8000
        user_config.bit_rate = 20  # Normally 10
        user_config.mod_type = "Duo-binary"  # Normally "NRZ"
        user_config.pattern = "PRBS-23"  # Normally PRBS-7
        user_config.Rdc = 2  # Normally 0.1876
        user_config.rin = 85  # Normally 100
        user_config.n_taps = 2  # Normally 5
        user_config.delta_t = 0.01  # Normally 0.1
        user_config.thresh = 5  # Normally 6
    with open(save_file, "w", encoding="UTF-8") as saved_config_file:
        yaml.dump(user_config, saved_config_file)

    dut.load_configuration(filepath_converter(save_file))

    # For everything saved in configuration, make sure they match.
    # All items should exist in both, so fail if one isn't found.
    for name in user_config.__dict__.keys():
        # These are handled differently so skip them.
        if name not in ["tx_taps", "tx_tap_tuners", "dfe_tap_tuners", "version", "date_created"]:
            # Test the values
            assert getattr(user_config, name) == getattr(dut, name)


@pytest.mark.usefixtures("dut")
def test_load_config_from_pickle(dut, tmp_path: Path):
    """Make sure that pybert can correctly load a pickle file."""

    # Manually save a configuration as pickle
    config = PyBertCfg(dut, "string_time", "test.test.test")
    save_file = tmp_path.joinpath("config.pybert_cfg")
    with open(save_file, "wb") as out_file:
        pickle.dump(config, out_file)
    TEST_PATTERN_LENGTH = 31

    # Modify the saved pickle file.
    with open(save_file, "rb") as saved_config_file:
        user_config = pickle.load(saved_config_file)
        user_config.pattern_len = TEST_PATTERN_LENGTH  # Normally, 127
    with open(save_file, "wb") as saved_config_file:
        pickle.dump(user_config, saved_config_file)

    dut.load_configuration(save_file)
    assert dut.pattern_len == TEST_PATTERN_LENGTH


@pytest.mark.usefixtures("dut")
def test_load_config_from_invalid(dut, tmp_path: Path):
    """
    When given an unsupported file suffix,
    an error message should be logged and an exception raised.
    """
    save_file = tmp_path.joinpath("config.json")
    save_file.touch()
    with pytest.raises(InvalidFileType) as excinfo:
        dut.load_configuration(save_file)

    assert "PyBERT does not support this file type." in str(excinfo.value)
    assert "This filetype is not currently supported." in dut.console_log


@pytest.mark.usefixtures("dut")
def test_new_eq_selector_traits_round_trip(dut, tmp_path: Path):
    """tx_eq_sel, rx_eq_sel, and ctle_sel survive a yaml save/load round-trip."""
    # tx_eq_sel stays "Native" without valid AMI — set it via internal path to bypass guard.
    dut.ctle_sel = "File"  # no guard on this one

    save_file = tmp_path / "eq_sel.yaml"
    dut.save_configuration(save_file)

    dut2 = PyBERT(run_simulation=False, gui=False)
    dut2.load_configuration(save_file)

    assert dut2.tx_eq_sel == "Native",  f"tx_eq_sel: expected 'Native', got {dut2.tx_eq_sel!r}"
    assert dut2.rx_eq_sel == "Native",  f"rx_eq_sel: expected 'Native', got {dut2.rx_eq_sel!r}"
    assert dut2.ctle_sel  == "File",    f"ctle_sel:  expected 'File',   got {dut2.ctle_sel!r}"


def test_ctle_sel_syncs_use_ctle_file():
    """Setting ctle_sel drives use_ctle_file."""
    dut = PyBERT(run_simulation=False, gui=False)
    assert dut.use_ctle_file is False
    dut.ctle_sel = "File"
    assert dut.use_ctle_file is True, "use_ctle_file should be True when ctle_sel='File'"
    dut.ctle_sel = "Native"
    assert dut.use_ctle_file is False, "use_ctle_file should be False when ctle_sel='Native'"


def test_eq_sel_case_insensitive_load(tmp_path: Path):
    """tx_eq_sel, rx_eq_sel, ctle_sel load correctly when written in any case in the config file."""
    dut = PyBERT(run_simulation=False, gui=False)
    save_file = tmp_path / "case_test.yaml"
    dut.save_configuration(save_file)

    with open(save_file, "r", encoding="UTF-8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg.tx_eq_sel = "native"     # lowercase
    cfg.rx_eq_sel = "NATIVE"     # uppercase
    cfg.ctle_sel  = "FILE"       # uppercase
    with open(save_file, "w", encoding="UTF-8") as f:
        yaml.dump(cfg, f)

    dut2 = PyBERT(run_simulation=False, gui=False)
    dut2.load_configuration(save_file)

    assert dut2.tx_eq_sel == "Native", f"Expected 'Native', got {dut2.tx_eq_sel!r}"
    assert dut2.rx_eq_sel == "Native", f"Expected 'Native', got {dut2.rx_eq_sel!r}"
    assert dut2.ctle_sel  == "File",   f"Expected 'File',   got {dut2.ctle_sel!r}"


def test_eq_sel_guard_no_ami():
    """tx_eq_sel/rx_eq_sel log a warning and leave tx_use_ami/rx_use_ami False when AMI not configured."""
    dut = PyBERT(run_simulation=False, gui=False)
    assert dut.tx_ami_valid is False
    assert dut.rx_ami_valid is False

    dut.tx_eq_sel = "IBIS-AMI"
    assert dut.tx_eq_sel  == "IBIS-AMI", "selector should accept the click (no snap-back)"
    assert dut.tx_use_ami is False,       "tx_use_ami must not be armed when guard fires"
    assert "IBIS-AMI mode" in dut.console_log, "guard should log a warning"

    dut.rx_eq_sel = "IBIS-AMI"
    assert dut.rx_eq_sel  == "IBIS-AMI", "selector should accept the click (no snap-back)"
    assert dut.rx_use_ami is False,       "rx_use_ami must not be armed when guard fires"


@pytest.mark.usefixtures("dut")
def test_load_results_from_pickle(dut, tmp_path: Path):
    """Make sure that pybert can correctly load a pickle file."""
    save_file = tmp_path.joinpath("config.pybert_data")
    dut.save_results(save_file)

    # Modify the saved pickle file.
    with open(save_file, "rb") as saved_results_file:
        user_results = pickle.load(saved_results_file)
        user_results.the_data.update_data({"chnl_h": np.array([1, 2, 3, 4])})
    with open(save_file, "wb") as saved_results_file:
        pickle.dump(user_results, saved_results_file)

    dut.load_results(save_file)
    # pybert doesn't directly reload the waveform back into the same plot.
    # instead if creates a reference plot to compare old vs. new.
    assert dut.plotdata.get_data("chnl_h_ref").size == 4
