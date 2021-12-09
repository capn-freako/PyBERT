"""
Simulation configuration data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   5 May 2017

This Python script provides a data structure for encapsulating the
simulation configuration data of a PyBERT instance. It was first
created, as a way to facilitate easier pickling, so that a particular
configuration could be saved and later restored.

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""


class PyBertCfg:
    """
    PyBERT simulation configuration data encapsulation class.

    This class is used to encapsulate that subset of the configuration
    data for a PyBERT instance, which is to be saved when the user
    clicks the "Save Config." button.
    """

    def __init__(self, the_PyBERT):
        """
        Copy just that subset of the supplied PyBERT instance's
        __dict__, which should be saved during pickling.
        """

        # Simulation Control
        self.bit_rate = the_PyBERT.bit_rate
        self.nbits = the_PyBERT.nbits
        self.pattern_len = the_PyBERT.pattern_len
        self.nspb = the_PyBERT.nspb
        self.eye_bits = the_PyBERT.eye_bits
        self.mod_type = list(the_PyBERT.mod_type)  # See Issue #95 and PR #98 (jdpatt)
        self.num_sweeps = the_PyBERT.num_sweeps
        self.sweep_num = the_PyBERT.sweep_num
        self.sweep_aves = the_PyBERT.sweep_aves
        self.do_sweep = the_PyBERT.do_sweep
        self.debug = the_PyBERT.debug

        # Channel Control
        self.use_ch_file = the_PyBERT.use_ch_file
        self.ch_file = the_PyBERT.ch_file
        self.impulse_length = the_PyBERT.impulse_length
        self.f_step = the_PyBERT.f_step
        self.Rdc = the_PyBERT.Rdc
        self.w0 = the_PyBERT.w0
        self.R0 = the_PyBERT.R0
        self.Theta0 = the_PyBERT.Theta0
        self.Z0 = the_PyBERT.Z0
        self.v0 = the_PyBERT.v0
        self.l_ch = the_PyBERT.l_ch

        # Tx
        self.vod = the_PyBERT.vod
        self.rs = the_PyBERT.rs
        self.cout = the_PyBERT.cout
        self.pn_mag = the_PyBERT.pn_mag
        self.pn_freq = the_PyBERT.pn_freq
        self.rn = the_PyBERT.rn
        tx_taps = []
        for tap in the_PyBERT.tx_taps:
            tx_taps.append((tap.enabled, tap.value))
        self.tx_taps = tx_taps
        self.tx_tap_tuners = []
        for tap in the_PyBERT.tx_tap_tuners:
            self.tx_tap_tuners.append((tap.enabled, tap.value))
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
        self.ctle_offset = the_PyBERT.ctle_offset
        self.ctle_mode = the_PyBERT.ctle_mode
        self.ctle_mode_tune = the_PyBERT.ctle_mode_tune
        self.ctle_offset_tune = the_PyBERT.ctle_offset_tune
        self.rx_use_ami = the_PyBERT.rx_use_ami
        self.rx_use_ts4 = the_PyBERT.rx_use_ts4
        self.rx_use_getwave = the_PyBERT.rx_use_getwave
        self.rx_ami_file = the_PyBERT.rx_ami_file
        self.rx_dll_file = the_PyBERT.rx_dll_file
        self.rx_ibis_file = the_PyBERT.rx_ibis_file
        self.rx_use_ibis = the_PyBERT.rx_use_ibis

        # DFE
        self.use_dfe = the_PyBERT.use_dfe
        self.use_dfe_tune = the_PyBERT.use_dfe_tune
        self.sum_ideal = the_PyBERT.sum_ideal
        self.decision_scaler = the_PyBERT.decision_scaler
        self.gain = the_PyBERT.gain
        self.n_ave = the_PyBERT.n_ave
        self.n_taps = the_PyBERT.n_taps
        self.sum_bw = the_PyBERT.sum_bw

        # CDR
        self.delta_t = the_PyBERT.delta_t
        self.alpha = the_PyBERT.alpha
        self.n_lock_ave = the_PyBERT.n_lock_ave
        self.rel_lock_tol = the_PyBERT.rel_lock_tol
        self.lock_sustain = the_PyBERT.lock_sustain

        # Analysis
        self.thresh = the_PyBERT.thresh
