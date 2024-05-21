"""Default controller definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
from time import perf_counter
from typing import Callable, Optional

import scipy.signal as sig
from chaco.api import Plot
from chaco.tools.api import PanTool, ZoomTool
from numpy import (  # type: ignore
    arange,
    argmax,
    array,
    convolve,
    correlate,
    diff,
    histogram,
    linspace,
    mean,
    repeat,
    resize,
    std,
    transpose,
    where,
    zeros,
)
from numpy.fft import rfft, irfft  # type: ignore
from numpy.random import normal  # type: ignore
from scipy.signal import iirfilter, lfilter
from scipy.interpolate import interp1d

from pybert.models.dfe import DFE
from pybert.utility import (
    calc_eye,
    calc_jitter,
    calc_resps,
    find_crossings,
    import_channel,
    make_bathtub,
    make_ctle,
    raised_cosine,
    run_ami_model,
    safe_log10,
    trim_impulse,
)

clock = perf_counter

DEBUG           = False
MIN_BATHTUB_VAL = 1.0e-18
gFc             = 1.0e6  # Corner frequency of high-pass filter used to model capacitive coupling of periodic noise.


def my_run_sweeps(self, is_thread_stopped: Optional[Callable[[], bool]] = None):
    """Runs the simulation sweeps.

    Args:
        self(PyBERT): Reference to an instance of the *PyBERT* class.
        is_thread_stopped: a function that is used to tell the simulation that the user
          has requested to stop/abort the simulation by setting the thread stop event.
    """

    sweep_aves = self.sweep_aves
    do_sweep = self.do_sweep
    tx_taps = self.tx_taps

    if do_sweep:
        # Assemble the list of desired values for each sweepable parameter.
        sweep_vals = []
        for tap in tx_taps:
            if tap.enabled:
                if tap.steps:
                    sweep_vals.append(list(arange(tap.min_val, tap.max_val, (tap.max_val - tap.min_val) / tap.steps)))
                else:
                    sweep_vals.append([tap.value])
            else:
                sweep_vals.append([0.0])
        # Run the sweep, using the lists assembled, above.
        sweeps = [
            [w, x, y, z] for w in sweep_vals[0] for x in sweep_vals[1] for y in sweep_vals[2] for z in sweep_vals[3]
        ]
        num_sweeps = sweep_aves * len(sweeps)
        self.num_sweeps = num_sweeps
        sweep_results = []
        sweep_num = 1
        for sweep in sweeps:
            for i in range(4):
                self.tx_taps[i].value = sweep[i]
            bit_errs = []
            for i in range(sweep_aves):
                self.sweep_num = sweep_num
                my_run_simulation(self, update_plots=False, aborted_sim=is_thread_stopped)
                bit_errs.append(self.bit_errs)
                sweep_num += 1
            sweep_results.append((sweep, mean(bit_errs), std(bit_errs)))
        self.sweep_results = sweep_results
    else:
        my_run_simulation(self, aborted_sim=is_thread_stopped)


# pylint: disable=too-many-locals,protected-access,too-many-branches,too-many-statements
def my_run_simulation(self, initial_run: bool = False, update_plots: bool = True,
                      aborted_sim: Optional[Callable[[], bool]] = None):
    """
    Runs the simulation.

    Args:
        self: Reference to an instance of the *PyBERT* class.

    Keyword Args:
        initial_run: If True, don't update the eye diagrams, since
            they haven't been created, yet.
            Default: False
        update_plots: If True, update the plots, after simulation
            completes. This option can be used by larger scripts, which
            import *pybert*, in order to avoid graphical back-end
            conflicts and speed up this function's execution time.
            Default: True
        aborted_sim: a function that is used to tell the simulation that the user
            has requested to stop/abort the simulation.

    Raises:
        RuntimeError: If the simulation is aborted by the user or cannot continue.

    Notes:
        1. When using IBIS-AMI models, we often need to scale the impulse response
            by the sample interval, or its inverse, because while IBIS-AMI models
            take the impulse response to have units: (V/s), PyBERT uses: (V/sample).
    """

    def _check_sim_status():
        """Checks the status of the simulation thread and if this simulation needs to stop."""
        if aborted_sim and aborted_sim():
            self.status = "Aborted Simulation"
            raise RuntimeError("Simulation aborted by User.")

    num_sweeps = self.num_sweeps
    sweep_num = self.sweep_num

    start_time = clock()
    self.status = f"Running channel...(sweep {sweep_num} of {num_sweeps})"

    if not self.seed:  # The user sets `seed` to zero to indicate that she wants new bits generated for each run.
        self.run_count += 1  # Force regeneration of bit stream.

    # Pull class variables into local storage, performing unit conversion where necessary.
    t = self.t
    t_irfft = self.t_irfft
    w = self.w
    bits = self.bits
    symbols = self.symbols
    ffe = self.ffe
    nbits = self.nbits
    nui = self.nui
    bit_rate = self.bit_rate * 1.0e9
    eye_bits = self.eye_bits
    eye_uis = self.eye_uis
    nspb = self.nspb
    nspui = self.nspui
    rn = self.rn
    pn_mag = self.pn_mag
    pn_freq = self.pn_freq * 1.0e6
    pattern = self.pattern_
    rx_bw = self.rx_bw * 1.0e9
    peak_freq = self.peak_freq * 1.0e9
    peak_mag = self.peak_mag
    ctle_offset = self.ctle_offset
    ctle_mode = self.ctle_mode
    delta_t = self.delta_t * 1.0e-12
    alpha = self.alpha
    ui = self.ui
    n_taps = self.n_taps
    gain = self.gain
    n_ave = self.n_ave
    decision_scaler = self.decision_scaler
    n_lock_ave = self.n_lock_ave
    rel_lock_tol = self.rel_lock_tol
    lock_sustain = self.lock_sustain
    bandwidth = self.sum_bw * 1.0e9
    rel_thresh = self.thresh
    mod_type = self.mod_type[0]
    impulse_length = self.impulse_length

    # Calculate misc. values.
    fs = bit_rate * nspb
    Ts = t[1]
    ts = Ts
    min_len =  30 * nspui
    max_len = 100 * nspui
    if impulse_length:
        min_len = max_len = impulse_length / ts

    # Generate the ideal over-sampled signal.
    #
    # Duo-binary is problematic, in that it requires convolution with the ideal duobinary
    # impulse response, in order to produce the proper ideal signal.
    x = repeat(symbols, nspui)
    ideal_signal = x
    if mod_type == 1:  # Handle duo-binary case.
        duob_h = array(([0.5] + [0.0] * (nspui - 1)) * 2)
        ideal_signal = convolve(x, duob_h)[: len(t)]

    # Calculate the channel response, as well as its (hypothetical)
    # solitary effect on the data, for plotting purposes only.
    try:
        split_time = clock()
        chnl_h = self.calc_chnl_h()
        _calc_chnl_time = clock() - split_time
        # Note: We're not using 'self.ideal_signal', because we rely on the system response to
        #       create the duobinary waveform. We only create it explicitly, above,
        #       so that we'll have an ideal reference for comparison.
        split_time = clock()
        chnl_out = convolve(x, chnl_h)[: len(t)]
        _conv_chnl_time = clock() - split_time
        if self.debug:
            self.log(f"Channel calculation time: {_calc_chnl_time}")
            self.log(f"Channel convolution time: {_conv_chnl_time}")
        self.channel_perf = nbits * nspb / (clock() - start_time)
    except Exception as err:
        self.status = f"Exception: channel: {err}"
        raise
    self.chnl_out = chnl_out

    _check_sim_status()
    split_time = clock()
    self.status = f"Running Tx...(sweep {sweep_num} of {num_sweeps})"

    # Calculate Tx output power dissipation.
    ffe_out = convolve(symbols, ffe)[: len(symbols)]
    if self.use_ch_file:
        self.rel_power = mean(ffe_out**2) / self.rs
    else:
        self.rel_power = mean(ffe_out**2) / self.Z0

    # Generate the uncorrelated periodic noise. (Assume capacitive coupling.)
    # Generate the ideal rectangular aggressor waveform.
    pn_period = 1.0 / pn_freq
    pn_samps = int(pn_period / Ts + 0.5)
    pn = zeros(pn_samps)
    pn[pn_samps // 2:] = pn_mag
    self.pn_period = pn_period
    self.pn_samps = pn_samps
    pn = resize(pn, len(x))
    # High pass filter it. (Simulating capacitive coupling.)
    (b, a) = iirfilter(2, gFc / (fs / 2), btype="highpass")
    pn = lfilter(b, a, pn)[: len(pn)]
    self.pn = pn

    noise = pn + normal(scale=rn, size=(len(x),))
    self.noise = noise

    # Tx and Rx modeling are not separable in all cases.
    # So, we model each of the 4 possible combinations explicitly.
    # For the purposes of tallying possible combinations,
    # AMI Init() and PyBERT native are equivalent,
    # as both rely on convolving w/ impulse responses.

    def get_ctle_h():
        "Return the impulse response of the PyBERT native CTLE model."
        if self.use_ctle_file:
            # FIXME: The new import_channel() implementation breaks this:  # pylint: disable=fixme
            ctle_h = import_channel(self.ctle_file, ts, self.f)
            if max(abs(ctle_h)) < 100.0:  # step response?
                ctle_h = diff(ctle_h)  # impulse response is derivative of step response.
            else:
                ctle_h *= ts  # Normalize to (V/sample)
            ctle_h.resize(len(t))
            ctle_H = rfft(ctle_h)  # ToDo: This needs interpolation first.  # pylint: disable=fixme
            # ctle_H *= sum(ctle_h) / ctle_H[0]
        else:
            _, ctle_H = make_ctle(rx_bw, peak_freq, peak_mag, w, ctle_mode, ctle_offset)
            ctle_h = irfft(raised_cosine(ctle_H))
            krnl = interp1d(t_irfft, ctle_h, bounds_error=False, fill_value=0)
            ctle_h = krnl(t)
            ctle_h *= t[1] / t_irfft[1]
        ctle_h, _ = trim_impulse(ctle_h, front_porch=False, min_len=min_len, max_len=max_len)
        return ctle_h

    try:
        if self.tx_use_ami and self.tx_use_getwave:
            tx_out, _, tx_h, tx_out_h, msg = run_ami_model(
                self.tx_dll_file, self._tx_cfg, True, ui, ts, chnl_h, x)
            self.log(f"Tx IBIS-AMI model initialization results:\n{msg}")
            rx_in = convolve(tx_out + noise, chnl_h)[:len(tx_out)]
            self.tx_perf = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = f"Running CTLE...(sweep {sweep_num} of {num_sweeps})"
            if self.rx_use_ami and self.rx_use_getwave:
                ctle_out, _, ctle_h, ctle_out_h, msg = run_ami_model(
                    self.rx_dll_file, self._rx_cfg, True, ui, ts, tx_out_h, convolve(tx_out, chnl_h))
                self.log(f"Rx IBIS-AMI model initialization results:\n{msg}")
            else:  # Rx is either AMI_Init() or PyBERT native.
                if self.rx_use_ami:  # Rx Init()
                    _, _, ctle_h, ctle_out_h, msg = run_ami_model(
                        self.rx_dll_file, self._rx_cfg, False, ui, ts, chnl_h, tx_out)
                    self.log(f"Rx IBIS-AMI model initialization results:\n{msg}")
                    ctle_out = convolve(tx_out, ctle_out_h)[:len(tx_out)]
                else:                # PyBERT native Rx
                    ctle_h = get_ctle_h()
                    ctle_out_h = convolve(ctle_h, tx_out_h)[:len(ctle_h)]
                    ctle_out = convolve(tx_out, convolve(ctle_h, chnl_h))[:len(tx_out)]
        else:  # Tx is either AMI_Init() or PyBERT native.
            if self.tx_use_ami:  # Tx is AMI_Init().
                rx_in, _, tx_h, tx_out_h, msg = run_ami_model(
                    self.tx_dll_file, self._tx_cfg, False, ui, ts, chnl_h, x)
                self.log(f"Tx IBIS-AMI model initialization results:\n{msg}")
                rx_in += noise
            else:                # Tx is PyBERT native.
                # Using `sum` to concatenate:
                tx_h = array(sum([[x] + list(zeros(nspui - 1)) for x in ffe], []))
                tx_h.resize(len(chnl_h), refcheck=False)  # "refcheck=False", to get around Tox failure.
                tx_out_h = convolve(tx_h, chnl_h)[: len(chnl_h)]
                rx_in = convolve(x, tx_out_h)[:len(x)] + noise
            self.tx_perf = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = f"Running CTLE...(sweep {sweep_num} of {num_sweeps})"
            if self.rx_use_ami and self.rx_use_getwave:
                ctle_out, _, ctle_h, ctle_out_h, msg = run_ami_model(
                    self.rx_dll_file, self._rx_cfg, True, ui, ts, tx_out_h, rx_in)
                self.log(f"Rx IBIS-AMI model initialization results:\n{msg}")
            else:  # Rx is either AMI_Init() or PyBERT native.
                if self.rx_use_ami:  # Rx Init()
                    ctle_out, _, ctle_h, ctle_out_h, msg = run_ami_model(
                        self.rx_dll_file, self._rx_cfg, False, ui, ts, tx_out_h, x)
                    self.log(f"Rx IBIS-AMI model initialization results:\n{msg}")
                    ctle_out += noise
                else:                # PyBERT native Rx
                    ctle_h = get_ctle_h()
                    ctle_out_h = convolve(ctle_h, tx_out_h)[:len(ctle_h)]
                    ctle_out = convolve(x + noise, ctle_out_h)[:len(x)]
    except Exception as err:
        self.status = f"Exception: {err}"
        raise

    # Calculate the remaining responses from the impulse responses.
    tx_s, tx_p, tx_H = calc_resps(t, tx_h, ui, t_fft=t_irfft)
    tx_out_s, tx_out_p, tx_out_H = calc_resps(t, tx_out_h, ui, t_fft=t_irfft)
    ctle_s, ctle_p, ctle_H = calc_resps(t, ctle_h, ui, t_fft=t_irfft)
    ctle_out_s, ctle_out_p, ctle_out_H = calc_resps(t, ctle_out_h, ui, t_fft=t_irfft)

    # Calculate convolutional delay.
    ctle_out.resize(len(t), refcheck=False)
    ctle_out_h_main_lobe = where(ctle_out_h >= max(ctle_out_h) / 2.0)[0]
    if ctle_out_h_main_lobe.size:
        conv_dly_ix = ctle_out_h_main_lobe[0]
    else:
        conv_dly_ix = int(self.chnl_dly // Ts)
    conv_dly = t[conv_dly_ix]

    # Stash needed intermediate results, as instance variables.
    self.tx_h = tx_h
    self.tx_s = tx_s
    self.tx_p = tx_p
    self.tx_H = tx_H
    self.tx_out_h = tx_out_h
    self.tx_out_s = tx_out_s
    self.tx_out_p = tx_out_p
    self.tx_out_H = tx_out_H
    self.ideal_signal = ideal_signal
    # self.tx_out = tx_out
    self.rx_in = rx_in
    self.ctle_h = ctle_h
    self.ctle_s = ctle_s
    self.ctle_p = ctle_p
    self.ctle_H = ctle_H
    self.ctle_out_h = ctle_out_h
    self.ctle_out_s = ctle_out_s
    self.ctle_out_p = ctle_out_p
    self.ctle_out_H = ctle_out_H
    self.ctle_out = ctle_out
    self.conv_dly = conv_dly
    self.conv_dly_ix = conv_dly_ix

    self.ctle_perf = nbits * nspb / (clock() - split_time)
    split_time = clock()
    self.status = "Running DFE/CDR...(sweep {sweep_num} of {num_sweeps})"

    _check_sim_status()

    # DFE output and incremental/cumulative responses.
    try:
        if self.use_dfe:
            _gain = gain
            _ideal = self.sum_ideal
        else:
            _gain = 0.0
            _ideal = True
        dfe = DFE(n_taps, _gain, delta_t, alpha, ui, nspui, decision_scaler, mod_type,
                  n_ave=n_ave, n_lock_ave=n_lock_ave, rel_lock_tol=rel_lock_tol,
                  lock_sustain=lock_sustain, bandwidth=bandwidth, ideal=_ideal)
        (dfe_out, tap_weights, ui_ests, clocks,
            lockeds, clock_times, bits_out) = dfe.run(t, ctle_out)
        dfe_out = array(dfe_out)
        dfe_out.resize(len(t))
        bits_out = array(bits_out)
        start_ix = len(bits_out) - eye_bits
        assert start_ix >= 0, "`start_ix` is negative!"
        end_ix = len(bits_out)
        auto_corr = (
            1.0 * correlate(bits_out[start_ix: end_ix], bits[start_ix: end_ix], mode="same") /  # noqa: W504
            sum(bits[start_ix: end_ix])
        )
        auto_corr = auto_corr[len(auto_corr) // 2:]
        self.auto_corr = auto_corr
        bit_dly = where(auto_corr == max(auto_corr))[0][0]
        bits_ref = bits[(nbits - eye_bits):]
        bits_tst = bits_out[(nbits + bit_dly - eye_bits):]
        if len(bits_ref) > len(bits_tst):
            bits_ref = bits_ref[: len(bits_tst)]
        elif len(bits_tst) > len(bits_ref):
            bits_tst = bits_tst[: len(bits_ref)]
        bit_errs = where(bits_tst ^ bits_ref)[0]
        self.bit_errs = len(bit_errs)

        dfe_h = array(
            [1.0] + list(zeros(nspb - 1)) +  # noqa: W504
            sum([[-x] + list(zeros(nspb - 1)) for x in tap_weights[-1]], []))  # sum as concat
        dfe_h.resize(len(ctle_out_h), refcheck=False)
        dfe_out_h = convolve(ctle_out_h, dfe_h)[: len(ctle_out_h)]

        # Calculate the remaining responses from the impulse responses.
        dfe_s, dfe_p, dfe_H = calc_resps(t, dfe_h, ui, t_fft=t_irfft)
        dfe_out_s, dfe_out_p, dfe_out_H = calc_resps(t, dfe_out_h, ui, t_fft=t_irfft)

        self.dfe_h = dfe_h
        self.dfe_s = dfe_s
        self.dfe_p = dfe_p
        self.dfe_H = dfe_H
        self.dfe_out_h = dfe_out_h
        self.dfe_out_s = dfe_out_s
        self.dfe_out_p = dfe_out_p
        self.dfe_out_H = dfe_out_H
        self.dfe_out = dfe_out

        self.dfe_perf = nbits * nspb / (clock() - split_time)
        split_time = clock()
        self.status = f"Analyzing jitter...(sweep {sweep_num} of {num_sweeps})"
    except Exception:
        self.status = "Exception: DFE"
        print(f"len(bits_out): {len(bits_out)}\nnbits: {nbits}\neye_bits: {eye_bits}")
        print(f"len(t): {len(t)}, len(ctle_out): {len(ctle_out)}")
        raise

    _check_sim_status()

    # Save local variables to class instance for state preservation, performing unit conversion where necessary.
    self.adaptation = tap_weights
    self.ui_ests = array(ui_ests) * 1.0e12  # (ps)
    self.clocks = clocks
    self.lockeds = lockeds
    self.clock_times = clock_times

    # Analyze the jitter.
    self.thresh_tx = array([])
    self.jitter_ext_tx = array([])
    self.jitter_tx = array([])
    self.jitter_spectrum_tx = array([])
    self.jitter_ind_spectrum_tx = array([])
    self.thresh_ctle = array([])
    self.jitter_ext_ctle = array([])
    self.jitter_ctle = array([])
    self.jitter_spectrum_ctle = array([])
    self.jitter_ind_spectrum_ctle = array([])
    self.thresh_dfe = array([])
    self.jitter_ext_dfe = array([])
    self.jitter_dfe = array([])
    self.jitter_spectrum_dfe = array([])
    self.jitter_ind_spectrum_dfe = array([])
    self.f_MHz_dfe = array([])
    self.jitter_rejection_ratio = array([])

    # The pattern length must be doubled in the duo-binary and PAM-4 cases anyway, because:
    #  - in the duo-binary case, the XOR pre-coding can invert every other pattern rep., and
    #  - in the PAM-4 case, the bits are taken in pairs to form the symbols and we start w/ an odd # of bits.
    # So, while it isn't strictly necessary, doubling it in the NRZ case as well provides a certain consistency.
    pattern_len = (pow(2, max(pattern)) - 1) * 2
    len_x_m1 = len(x) - 1
    xing_min_t = (nui - eye_uis) * ui

    def eye_xings(xings, ofst=0):
        """
        Return crossings from that portion of the signal used to generate the eye.

        Args:
            xings([float]): List of crossings.

        KeywordArgs:
            ofst(float): Time offset to be subtracted from all crossings.

        Returns:
            [float]: Selected crossings, offset and eye-start corrected.
        """
        _xings = array(xings) - ofst
        return _xings[where(_xings > xing_min_t)] - xing_min_t

    try:
        # - ideal
        ideal_xings = find_crossings(t, ideal_signal, decision_scaler, mod_type=mod_type)
        self.ideal_xings = ideal_xings
        ideal_xings_jit = eye_xings(ideal_xings)

        # - channel output
        ofst = (argmax(sig.correlate(chnl_out, x)) - len_x_m1) * Ts
        actual_xings = find_crossings(t, chnl_out, decision_scaler, mod_type=mod_type)
        actual_xings_jit = eye_xings(actual_xings, ofst)
        (
            tie,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            pjDD,
            rjDD,
            tie_ind,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.t_jitter  = t_jitter
        self.isi_chnl  = isi
        self.dcd_chnl  = dcd
        self.pj_chnl   = pj
        self.rj_chnl   = rj
        self.pjDD_chnl = pjDD
        self.rjDD_chnl = rjDD
        self.thresh_chnl = thresh
        self.jitter_chnl = hist
        self.jitter_ext_chnl = hist_synth
        self.jitter_bins = bin_centers
        self.jitter_spectrum_chnl = jitter_spectrum
        self.jitter_ind_spectrum_chnl = jitter_ind_spectrum
        self.f_MHz = array(spectrum_freqs) * 1.0e-6
        self.ofst_chnl = ofst
        self.tie_chnl = tie
        self.tie_ind_chnl = tie_ind

        # - Tx output
        ofst = (argmax(sig.correlate(rx_in, x)) - len_x_m1) * Ts
        actual_xings = find_crossings(t, rx_in, decision_scaler, mod_type=mod_type)
        actual_xings_jit = eye_xings(actual_xings, ofst)
        (
            tie,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            pjDD,
            rjDD,
            tie_ind,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh, dbg_obj=self)
        self.isi_tx  = isi
        self.dcd_tx  = dcd
        self.pj_tx   = pj
        self.rj_tx   = rj
        self.pjDD_tx = pjDD
        self.rjDD_tx = rjDD
        self.thresh_tx = thresh
        self.jitter_tx = hist
        self.jitter_ext_tx = hist_synth
        self.jitter_centers_tx = bin_centers
        self.jitter_spectrum_tx = jitter_spectrum
        self.jitter_ind_spectrum_tx = jitter_ind_spectrum
        self.jitter_freqs_tx = spectrum_freqs
        self.t_jitter_tx = t_jitter
        self.tie_tx = tie
        self.tie_ind_tx = tie_ind

        # - CTLE output
        ofst = (argmax(sig.correlate(ctle_out, x)) - len_x_m1) * Ts
        actual_xings = find_crossings(t, ctle_out, decision_scaler, mod_type=mod_type)
        actual_xings_jit = eye_xings(actual_xings, ofst)
        (
            tie,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            pjDD,
            rjDD,
            tie_ind,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.isi_ctle  = isi
        self.dcd_ctle  = dcd
        self.pj_ctle   = pj
        self.rj_ctle   = rj
        self.pjDD_ctle = pjDD
        self.rjDD_ctle = rjDD
        self.thresh_ctle = thresh
        self.jitter_ctle = hist
        self.jitter_ext_ctle = hist_synth
        self.jitter_spectrum_ctle = jitter_spectrum
        self.jitter_ind_spectrum_ctle = jitter_ind_spectrum
        self.tie_ctle = tie
        self.tie_ind_ctle = tie_ind

        # - DFE output
        ofst = (argmax(sig.correlate(dfe_out, x)) - len_x_m1) * Ts
        actual_xings = find_crossings(t, dfe_out, decision_scaler, mod_type=mod_type)
        actual_xings_jit = eye_xings(actual_xings, ofst)
        (
            tie,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            pjDD,
            rjDD,
            tie_ind,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.isi_dfe  = isi
        self.dcd_dfe  = dcd
        self.pj_dfe   = pj
        self.rj_dfe   = rj
        self.pjDD_dfe = pjDD
        self.rjDD_dfe = rjDD
        self.thresh_dfe = thresh
        self.jitter_dfe = hist
        self.jitter_ext_dfe = hist_synth
        self.jitter_spectrum_dfe = jitter_spectrum
        self.jitter_ind_spectrum_dfe = jitter_ind_spectrum
        self.tie_dfe = tie
        self.tie_ind_dfe = tie_ind
        self.f_MHz_dfe = array(spectrum_freqs) * 1.0e-6
        dfe_spec = self.jitter_spectrum_dfe
        self.jitter_rejection_ratio = zeros(len(dfe_spec))

        self.jitter_perf = nbits * nspb / (clock() - split_time)
        self.total_perf = nbits * nspb / (clock() - start_time)
        split_time = clock()
        self.status = f"Updating plots...(sweep {sweep_num} of {num_sweeps})"
    except Exception:
        # if update_plots:
        #     update_results(self)
        #     if not initial_run:
        #         update_eyes(self)
        self.status = "Exception: jitter"
        raise

    _check_sim_status()
    # Update plots.
    try:
        if update_plots:
            update_results(self)
            if not initial_run:
                update_eyes(self)

        self.plotting_perf = nbits * nspb / (clock() - split_time)
        self.status = "Ready."
    except Exception:
        self.status = "Exception: plotting"
        raise


# Plot updating
# pylint: disable=too-many-locals,protected-access,too-many-statements
def update_results(self):
    """Updates all plot data used by GUI.

    Args:
        self(PyBERT): Reference to an instance of the *PyBERT* class.
    """

    # Copy globals into local namespace.
    ui = self.ui
    samps_per_ui = self.nspui
    eye_uis = self.eye_uis
    num_ui = self.nui
    clock_times = self.clock_times
    f = self.f
    t = self.t
    t_ns = self.t_ns
    t_ns_chnl = self.t_ns_chnl
    n_taps = self.n_taps
    t_irfft = self.t_irfft

    Ts = t[1]
    ignore_until = (num_ui - eye_uis) * ui
    ignore_samps = (num_ui - eye_uis) * samps_per_ui

    # Misc.
    f_GHz = f / 1.0e9
    len_f_GHz = len(f_GHz)
    len_t = len(t_ns)
    self.plotdata.set_data("f_GHz", f_GHz[1:])
    self.plotdata.set_data("t_ns", t_ns)
    self.plotdata.set_data("t_ns_chnl", t_ns_chnl)
    self.plotdata.set_data("t_ns_irfft", t_irfft * 1e9)

    # DFE.
    tap_weights = transpose(array(self.adaptation))
    i = 1
    for tap_weight in tap_weights:  # pylint: disable=undefined-loop-variable
        self.plotdata.set_data(f"tap{i}_weights", tap_weight)
        i += 1
    self.plotdata.set_data("tap_weight_index", list(range(len(tap_weight))))  # pylint: disable=undefined-loop-variable
    if self._old_n_taps != n_taps:  # pylint: disable=undefined-loop-variable
        new_plot = Plot(
            self.plotdata,
            auto_colors=["red", "orange", "yellow", "green", "blue", "purple"],
            padding_left=75,
        )
        for i in range(self.n_taps):
            new_plot.plot(
                ("tap_weight_index", f"tap{i + 1}_weights"),
                type="line",
                name=f"tap{i + 1}",
                color="auto")
        new_plot.title = "DFE Adaptation"
        new_plot.tools.append(PanTool(new_plot, constrain=True, constrain_key=None, constrain_direction="x"))
        zoom9 = ZoomTool(new_plot, tool_mode="range", axis="index", always_on=False)
        new_plot.overlays.append(zoom9)
        new_plot.legend.visible = True
        new_plot.legend.align = "ul"
        self.plots_dfe.remove(self._dfe_plot)
        self.plots_dfe.insert(1, new_plot)
        self._dfe_plot = new_plot
        self._old_n_taps = n_taps

    clock_pers = diff(clock_times)
    lockedsTrue = where(self.lockeds)[0]
    if lockedsTrue.any():
        start_t = t[lockedsTrue[0]]
    else:
        start_t = 0
    start_ix = where(array(clock_times) > start_t)[0][0]
    (bin_counts, bin_edges) = histogram(clock_pers[start_ix:], bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    clock_spec = rfft(clock_pers[start_ix:])
    clock_spec = abs(clock_spec[: len(clock_spec) // 2])
    spec_freqs = arange(len(clock_spec)) / (2.0 * len(clock_spec))  # In this case, fNyquist = half the bit rate.
    clock_spec /= clock_spec[1:].mean()  # Normalize the mean non-d.c. value to 0 dB.
    self.plotdata.set_data("clk_per_hist_bins", bin_centers * 1.0e12)  # (ps)
    self.plotdata.set_data("clk_per_hist_vals", bin_counts)
    self.plotdata.set_data("clk_spec", 10.0 * safe_log10(clock_spec[1:]))  # Omit the d.c. value.
    self.plotdata.set_data("clk_freqs", spec_freqs[1:])
    self.plotdata.set_data("dfe_out", self.dfe_out)
    self.plotdata.set_data("ui_ests", self.ui_ests)
    self.plotdata.set_data("clocks", self.clocks)
    self.plotdata.set_data("lockeds", self.lockeds)

    # Impulse responses
    self.plotdata.set_data("chnl_h", self.chnl_h * 1.0e-9 / Ts)  # Re-normalize to (V/ns), for plotting.
    # self.plotdata.set_data("chnl_h", self.chnl_h_raw * 1.0e-9 / Ts)  # Re-normalize to (V/ns), for plotting.
    self.plotdata.set_data("tx_h", self.tx_h * 1.0e-9 / Ts)
    self.plotdata.set_data("tx_out_h", self.tx_out_h * 1.0e-9 / Ts)
    self.plotdata.set_data("ctle_h", self.ctle_h * 1.0e-9 / Ts)
    self.plotdata.set_data("ctle_out_h", self.ctle_out_h * 1.0e-9 / Ts)
    self.plotdata.set_data("dfe_h", self.dfe_h * 1.0e-9 / Ts)
    self.plotdata.set_data("dfe_out_h", self.dfe_out_h * 1.0e-9 / Ts)

    # Step responses
    self.plotdata.set_data("chnl_s", self.chnl_s)
    self.plotdata.set_data("tx_s", self.tx_s)
    self.plotdata.set_data("tx_out_s", self.tx_out_s)
    self.plotdata.set_data("ctle_s", self.ctle_s)
    self.plotdata.set_data("ctle_out_s", self.ctle_out_s)
    self.plotdata.set_data("dfe_s", self.dfe_s)
    self.plotdata.set_data("dfe_out_s", self.dfe_out_s)

    # Pulse responses
    self.plotdata.set_data("chnl_p", self.chnl_p)
    self.plotdata.set_data("tx_out_p", self.tx_out_p)
    self.plotdata.set_data("ctle_out_p", self.ctle_out_p)
    self.plotdata.set_data("dfe_out_p", self.dfe_out_p)

    # Outputs
    self.plotdata.set_data("ideal_signal", self.ideal_signal[:len_t])
    self.plotdata.set_data("chnl_out", self.chnl_out[:len_t])
    self.plotdata.set_data("tx_out", self.rx_in[:len_t])
    self.plotdata.set_data("ctle_out", self.ctle_out[:len_t])
    self.plotdata.set_data("dfe_out", self.dfe_out[:len_t])

    # Frequency responses
    self.plotdata.set_data("chnl_H_raw", 20.0 * safe_log10(abs(self.chnl_H_raw[1:len_f_GHz])))
    self.plotdata.set_data("chnl_H", 20.0 * safe_log10(abs(self.chnl_H[1:len_f_GHz])))
    self.plotdata.set_data("chnl_trimmed_H", 20.0 * safe_log10(abs(self.chnl_trimmed_H[1:len_f_GHz])))
    self.plotdata.set_data("tx_H", 20.0 * safe_log10(abs(self.tx_H[1:])))
    self.plotdata.set_data("tx_out_H", 20.0 * safe_log10(abs(self.tx_out_H[1:len_f_GHz])))
    self.plotdata.set_data("ctle_H", 20.0 * safe_log10(abs(self.ctle_H[1:len_f_GHz])))
    self.plotdata.set_data("ctle_out_H", 20.0 * safe_log10(abs(self.ctle_out_H[1:len_f_GHz])))
    self.plotdata.set_data("dfe_H", 20.0 * safe_log10(abs(self.dfe_H[1:len_f_GHz])))
    self.plotdata.set_data("dfe_out_H", 20.0 * safe_log10(abs(self.dfe_out_H[1:len_f_GHz])))

    # Jitter distributions
    jitter_chnl = self.jitter_chnl  # These are used again in bathtub curve generation, below.
    jitter_tx   = self.jitter_tx
    jitter_ctle = self.jitter_ctle
    jitter_dfe  = self.jitter_dfe
    jitter_bins = self.jitter_bins
    self.plotdata.set_data("jitter_bins", array(self.jitter_bins)  * 1e12)
    self.plotdata.set_data("jitter_chnl",     jitter_chnl          * 1e-12)  # PDF (/ps)
    self.plotdata.set_data("jitter_ext_chnl", self.jitter_ext_chnl * 1e-12)
    self.plotdata.set_data("jitter_tx",       jitter_tx            * 1e-12)
    self.plotdata.set_data("jitter_ext_tx",   self.jitter_ext_tx   * 1e-12)
    self.plotdata.set_data("jitter_ctle",     jitter_ctle          * 1e-12)
    self.plotdata.set_data("jitter_ext_ctle", self.jitter_ext_ctle * 1e-12)
    self.plotdata.set_data("jitter_dfe",      jitter_dfe           * 1e-12)
    self.plotdata.set_data("jitter_ext_dfe",  self.jitter_ext_dfe  * 1e-12)

    # Jitter spectrums
    log10_ui = safe_log10(ui)
    self.plotdata.set_data("f_MHz", self.f_MHz[1:])
    self.plotdata.set_data("f_MHz_dfe", self.f_MHz_dfe[1:])
    self.plotdata.set_data("jitter_spectrum_chnl", 10.0 * (safe_log10(self.jitter_spectrum_chnl[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_chnl", 10.0 * (safe_log10(self.jitter_ind_spectrum_chnl[1:]) - log10_ui))
    self.plotdata.set_data("thresh_chnl", 10.0 * (safe_log10(self.thresh_chnl[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_tx", 10.0 * (safe_log10(self.jitter_spectrum_tx[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_tx", 10.0 * (safe_log10(self.jitter_ind_spectrum_tx[1:]) - log10_ui))
    self.plotdata.set_data("thresh_tx", 10.0 * (safe_log10(self.thresh_tx[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_ctle", 10.0 * (safe_log10(self.jitter_spectrum_ctle[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_ctle", 10.0 * (safe_log10(self.jitter_ind_spectrum_ctle[1:]) - log10_ui))
    self.plotdata.set_data("thresh_ctle", 10.0 * (safe_log10(self.thresh_ctle[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_dfe", 10.0 * (safe_log10(self.jitter_spectrum_dfe[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_dfe", 10.0 * (safe_log10(self.jitter_ind_spectrum_dfe[1:]) - log10_ui))
    self.plotdata.set_data("thresh_dfe", 10.0 * (safe_log10(self.thresh_dfe[1:]) - log10_ui))
    self.plotdata.set_data("jitter_rejection_ratio", self.jitter_rejection_ratio[1:])

    # Bathtubs
    bathtub_chnl, (_, _) = make_bathtub(
        jitter_bins, jitter_chnl, min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rj_chnl, extrap=True)
    bathtub_tx,   (_,  _) = make_bathtub(
        jitter_bins, jitter_tx,   min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rj_tx,   extrap=True)
    bathtub_ctle, (_, _) = make_bathtub(
        jitter_bins, jitter_ctle, min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rj_ctle, extrap=True)
    bathtub_dfe,  (_,  _) = make_bathtub(
        jitter_bins, jitter_dfe,  min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rj_dfe,  extrap=True)
    self.plotdata.set_data("bathtub_chnl", safe_log10(bathtub_chnl))
    self.plotdata.set_data("bathtub_tx",   safe_log10(bathtub_tx))
    self.plotdata.set_data("bathtub_ctle", safe_log10(bathtub_ctle))
    self.plotdata.set_data("bathtub_dfe",  safe_log10(bathtub_dfe))

    # Eyes
    width = 2 * samps_per_ui
    xs = linspace(-ui * 1.0e12, ui * 1.0e12, width)
    height = 100
    y_max = 1.1 * max(abs(array(self.chnl_out)))
    eye_chnl = calc_eye(ui, samps_per_ui, height, self.chnl_out[ignore_samps:], y_max)
    y_max = 1.1 * max(abs(array(self.rx_in)))
    eye_tx = calc_eye(ui, samps_per_ui, height, self.rx_in[ignore_samps:], y_max)
    y_max = 1.1 * max(abs(array(self.ctle_out)))
    eye_ctle = calc_eye(ui, samps_per_ui, height, self.ctle_out[ignore_samps:], y_max)
    i = 0
    while clock_times[i] <= ignore_until:
        i += 1
        assert i < len(clock_times), "ERROR: Insufficient coverage in 'clock_times' vector."
    y_max = 1.1 * max(abs(array(self.dfe_out)))
    eye_dfe = calc_eye(ui, samps_per_ui, height, self.dfe_out, y_max, clock_times[i:])
    self.plotdata.set_data("eye_index", xs)
    self.plotdata.set_data("eye_chnl", eye_chnl)
    self.plotdata.set_data("eye_tx", eye_tx)
    self.plotdata.set_data("eye_ctle", eye_ctle)
    self.plotdata.set_data("eye_dfe", eye_dfe)


def update_eyes(self):
    """Update the heat plots representing the eye diagrams.

    Args:
        self(PyBERT): Reference to an instance of the *PyBERT* class.
    """

    ui = self.ui
    samps_per_ui = self.nspui

    width = 2 * samps_per_ui
    height = 100
    xs = linspace(-ui * 1.0e12, ui * 1.0e12, width)

    y_max = 1.1 * max(abs(array(self.chnl_out)))
    ys = linspace(-y_max, y_max, height)
    self.plots_eye.components[0].components[0].index.set_data(xs, ys)
    self.plots_eye.components[0].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[0].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[0].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[0].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[0].invalidate_draw()

    y_max = 1.1 * max(abs(array(self.rx_in)))
    ys = linspace(-y_max, y_max, height)
    self.plots_eye.components[1].components[0].index.set_data(xs, ys)
    self.plots_eye.components[1].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[1].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[1].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[1].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[1].invalidate_draw()

    y_max = 1.1 * max(abs(array(self.dfe_out)))
    ys = linspace(-y_max, y_max, height)
    self.plots_eye.components[3].components[0].index.set_data(xs, ys)
    self.plots_eye.components[3].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[3].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[3].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[3].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[3].invalidate_draw()

    self.plots_eye.components[2].components[0].index.set_data(xs, ys)
    self.plots_eye.components[2].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[2].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[2].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[2].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[2].invalidate_draw()

    self.plots_eye.request_redraw()
