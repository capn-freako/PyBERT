"""Simulation configuration data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   5 May 2017

This Python script provides a data structure for encapsulating the
simulation configuration data of a PyBERT instance. It was first
created, as a way to facilitate easier pickling, so that a particular
configuration could be saved and later restored.

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""
import pickle
import warnings
from pathlib import Path
from typing import Union

import yaml


class InvalidFileType(Exception):
    """Raised when a filetype that isn't supported is used when trying to load
    or save files.."""


# These are different for now to allow users to "upgrade" their configuration file.

CONFIG_LOAD_WILDCARD = "|".join(
    [
        "Yaml Config (*.yaml;*.yml)|*.yaml;*.yml",
        "Pickle Config (*.pybert_cfg)|*.pybert_cfg",
        "All files (*)|*",
    ]
)
"""This sets the supported file types in the GUI's loading dialog."""

CONFIG_SAVE_WILDCARD = "|".join(
    [
        "Yaml Config (*.yaml;*.yml)|*.yaml;*.yml",
        "All files (*)|*",
    ]
)
"""This sets the supported file types in the GUI's save-as dialog."""


class PyBertCfg:  # pylint: disable=too-many-instance-attributes
    """PyBERT simulation configuration data encapsulation class.

    This class is used to encapsulate that subset of the configuration
    data for a PyBERT instance, which is to be saved when the user
    clicks the "Save Config." button.
    """

    def __init__(self, the_PyBERT, date_created: str, version: str):  # pylint: disable=too-many-statements
        """Copy just that subset of the supplied PyBERT instance's __dict__,
        which should be saved."""

        # Generic Information
        self.date_created = date_created
        self.version = version

        # Simulation Control
        self.bit_rate = the_PyBERT.bit_rate
        self.nbits = the_PyBERT.nbits
        self.pattern = the_PyBERT.pattern
        self.seed = the_PyBERT.seed
        self.nspui = the_PyBERT.nspui
        self.eye_bits = the_PyBERT.eye_bits
        self.mod_type = list(the_PyBERT.mod_type)  # See Issue #95 and PR #98 (jdpatt)
        self.debug = the_PyBERT.debug
        self.f_max = the_PyBERT.f_max
        self.f_step = the_PyBERT.f_step

        # Channel Control
        self.use_ch_file = the_PyBERT.use_ch_file
        self.ch_file = the_PyBERT.ch_file
        self.impulse_length = the_PyBERT.impulse_length
        self.Rdc = the_PyBERT.Rdc
        self.w0 = the_PyBERT.w0
        self.R0 = the_PyBERT.R0
        self.Theta0 = the_PyBERT.Theta0
        self.Z0 = the_PyBERT.Z0
        self.v0 = the_PyBERT.v0
        self.l_ch = the_PyBERT.l_ch
        self.renumber = the_PyBERT.renumber
        self.use_window = the_PyBERT.use_window

        # Tx
        self.vod = the_PyBERT.vod
        self.rs = the_PyBERT.rs
        self.cout = the_PyBERT.cout
        self.pn_mag = the_PyBERT.pn_mag
        self.pn_freq = the_PyBERT.pn_freq
        self.rn = the_PyBERT.rn
        tx_taps = []
        for tap in the_PyBERT.tx_taps:
            tx_taps.append((tap.enabled, tap.value, tap.min_val, tap.max_val))
        self.tx_taps = tx_taps
        self.tx_tap_tuners = []
        for tap in the_PyBERT.tx_tap_tuners:
            self.tx_tap_tuners.append((tap.enabled, tap.pos, tap.min_val, tap.max_val, tap.step))
        self.tx_use_ami = the_PyBERT.tx_use_ami
        self.tx_use_ts4 = the_PyBERT.tx_use_ts4
        self.tx_use_getwave = the_PyBERT.tx_use_getwave
        self.tx_ami_file = the_PyBERT.tx_ami_file
        self.tx_dll_file = the_PyBERT.tx_dll_file
        self.tx_ibis_file = the_PyBERT.tx_ibis_file
        self.tx_use_ibis = the_PyBERT.tx_use_ibis

        # Rx
        self.rin = the_PyBERT.rin
        self.cin = the_PyBERT.cin
        self.cac = the_PyBERT.cac
        self.use_ctle_file = the_PyBERT.use_ctle_file
        self.ctle_file = the_PyBERT.ctle_file
        self.rx_bw = the_PyBERT.rx_bw
        self.peak_freq = the_PyBERT.peak_freq
        self.peak_mag = the_PyBERT.peak_mag
        self.ctle_enable = the_PyBERT.ctle_enable
        self.rx_use_ami = the_PyBERT.rx_use_ami
        self.rx_use_ts4 = the_PyBERT.rx_use_ts4
        self.rx_use_getwave = the_PyBERT.rx_use_getwave
        self.rx_ami_file = the_PyBERT.rx_ami_file
        self.rx_dll_file = the_PyBERT.rx_dll_file
        self.rx_ibis_file = the_PyBERT.rx_ibis_file
        self.rx_use_ibis = the_PyBERT.rx_use_ibis
        self.rx_use_viterbi = the_PyBERT.rx_use_viterbi
        self.rx_viterbi_symbols = the_PyBERT.rx_viterbi_symbols

        # DFE
        self.sum_ideal = the_PyBERT.sum_ideal
        self.decision_scaler = the_PyBERT.decision_scaler
        self.gain = the_PyBERT.gain
        self.n_ave = the_PyBERT.n_ave
        self.sum_bw = the_PyBERT.sum_bw
        self.use_agc = the_PyBERT.use_agc

        # CDR
        self.delta_t = the_PyBERT.delta_t
        self.alpha = the_PyBERT.alpha
        self.n_lock_ave = the_PyBERT.n_lock_ave
        self.rel_lock_tol = the_PyBERT.rel_lock_tol
        self.lock_sustain = the_PyBERT.lock_sustain

        # Analysis
        self.thresh = the_PyBERT.thresh

        # Optimization
        self.rx_bw_tune = the_PyBERT.rx_bw_tune
        self.peak_freq_tune = the_PyBERT.peak_freq_tune
        self.peak_mag_tune = the_PyBERT.peak_mag_tune
        self.min_mag_tune = the_PyBERT.min_mag_tune
        self.max_mag_tune = the_PyBERT.max_mag_tune
        self.step_mag_tune = the_PyBERT.step_mag_tune
        self.ctle_enable_tune = the_PyBERT.ctle_enable_tune
        self.dfe_tap_tuners = []
        for tap in the_PyBERT.dfe_tap_tuners:
            self.dfe_tap_tuners.append((tap.enabled, tap.min_val, tap.max_val))

    @staticmethod
    def load_from_file(filepath: Union[str, Path], pybert):  # pylint: disable=too-many-branches
        """Apply all of the configuration settings to the pybert instance.

        Confirms that the file actually exists, is the correct extension and
        attempts to set the values back in pybert.

        Args:
            filepath: The full filepath including the extension to save too.
            pybert: instance of the main app
        """
        filepath = Path(filepath)  # incase a string was passed convert to a path.

        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        # If its a valid extension load it.
        if filepath.suffix in [".yaml", ".yml"]:
            with open(filepath, "r", encoding="UTF-8") as yaml_file:
                user_config = yaml.load(yaml_file, Loader=yaml.Loader)
        elif filepath.suffix == ".pybert_cfg":
            warnings.warn(
                "Using pickle for configuration is not suggested and will be removed in a later release.",
                DeprecationWarning,
                stacklevel=2,
            )
            with open(filepath, "rb") as pickle_file:
                user_config = pickle.load(pickle_file)
        else:
            raise InvalidFileType("Pybert does not support this file type.")

        # Right now the loads deserialize back into a `PyBertCfg` class.
        if not isinstance(user_config, PyBertCfg):
            raise ValueError("The data structure read in is NOT of type: PyBertCfg!")

        # Actually load values back into pybert using `setattr`.
        for prop, value in vars(user_config).items():
            if prop == "tx_taps":
                for count, (enabled, val, min_val, max_val) in enumerate(value):
                    setattr(pybert.tx_taps[count], "enabled", enabled)
                    setattr(pybert.tx_taps[count], "value", val)
                    setattr(pybert.tx_taps[count], "min_val", min_val)
                    setattr(pybert.tx_taps[count], "max_val", max_val)
            elif prop == "tx_tap_tuners":
                for count, (enabled, pos, min_val, max_val, step) in enumerate(value):
                    setattr(pybert.tx_tap_tuners[count], "enabled", enabled)
                    setattr(pybert.tx_tap_tuners[count], "pos", pos)
                    setattr(pybert.tx_tap_tuners[count], "min_val", min_val)
                    setattr(pybert.tx_tap_tuners[count], "max_val", max_val)
                    setattr(pybert.tx_tap_tuners[count], "step", step)
            elif prop == "dfe_tap_tuners":
                for count, (enabled, min_val, max_val) in enumerate(value):
                    setattr(pybert.dfe_tap_tuners[count], "enabled", enabled)
                    setattr(pybert.dfe_tap_tuners[count], "min_val", min_val)
                    setattr(pybert.dfe_tap_tuners[count], "max_val", max_val)
            elif prop in ("version", "date_created"):
                pass  # Just including it for some good housekeeping.  Not currently used.
            else:
                setattr(pybert, prop, value)

    def save(self, filepath: Union[str, Path]):
        """Save out pybert's current configuration to a file.

        The extension must match a yaml file extension or it will still raise
        an invalid file type.  Additional filetypes can be added/supported by
        just adding another if statement and adding to `CONFIG_FILEDIALOG_WILDCARD`.

        Args:
            filepath: The full filepath including the extension to save too.
        """
        filepath = Path(filepath)  # incase a string was passed convert to a path.

        if filepath.suffix in [".yaml", ".yml"]:
            with open(filepath, "w", encoding="UTF-8") as yaml_file:
                yaml.dump(self, yaml_file, indent=4, sort_keys=False)
        else:
            raise InvalidFileType("Pybert does not support this file type.")
