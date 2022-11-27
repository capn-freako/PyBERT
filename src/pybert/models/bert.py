"""Default controller definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
from time import perf_counter

clock = perf_counter

import scipy.signal as sig
from numpy import (
    arange,
    argmax,
    array,
    convolve,
    correlate,
    cumsum,
    diff,
    histogram,
    linspace,
    mean,
    ones,
    pad,
    repeat,
    resize,
    std,
    transpose,
    where,
    zeros,
)
from numpy.fft import fft, irfft
from numpy.random import normal
from pyibisami.ami.model import AMIModel, AMIModelInitializer
from scipy.signal import iirfilter, lfilter

from chaco.api import Plot
from chaco.tools.api import PanTool, ZoomTool
from pybert.models.dfe import DFE
from pybert.utility import (
    calc_eye,
    calc_jitter,
    find_crossings,
    getwave_step_resp,
    import_channel,
    make_ctle,
    safe_log10,
    trim_impulse,
)

DEBUG = False
MIN_BATHTUB_VAL = 1.0e-18

gFc = 1.0e6  # Corner frequency of high-pass filter used to model capacitive coupling of periodic noise.


def my_run_sweeps(self):
    """Runs the simulation sweeps.

    Args:
        self(PyBERT): Reference to an instance of the *PyBERT* class.
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
                my_run_simulation(self, update_plots=False)
                bit_errs.append(self.bit_errs)
                sweep_num += 1
            sweep_results.append((sweep, mean(bit_errs), std(bit_errs)))
        self.sweep_results = sweep_results
    else:
        my_run_simulation(self)


def my_run_simulation(self, initial_run=False, update_plots=True):
    """Runs the simulation.

    Args:
        self(PyBERT): Reference to an instance of the *PyBERT* class.
        initial_run(Bool): If True, don't update the eye diagrams, since
            they haven't been created, yet.
            (Optional; default = False.)
        update_plots(Bool): If True, update the plots, after simulation
            completes. This option can be used by larger scripts, which
            import *pybert*, in order to avoid graphical back-end
            conflicts and speed up this function's execution time.
            (Optional; default = True.)
    """
    num_sweeps = self.num_sweeps
    sweep_num = self.sweep_num

    start_time = clock()
    self.status = "Running channel...(sweep %d of %d)" % (sweep_num, num_sweeps)

    if not self.seed:  # The user sets `seed` to zero to indicate that she wants new bits generated for each run.
        self.run_count += 1  # Force regeneration of bit stream.

    # Pull class variables into local storage, performing unit conversion where necessary.
    t = self.t
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

    try:
        # Calculate misc. values.
        fs = bit_rate * nspb
        Ts = t[1]
        ts = Ts

        # Generate the ideal over-sampled signal.
        #
        # Duo-binary is problematic, in that it requires convolution with the ideal duobinary
        # impulse response, in order to produce the proper ideal signal.
        x = repeat(symbols, nspui)
        self.x = x
        if mod_type == 1:  # Handle duo-binary case.
            duob_h = array(([0.5] + [0.0] * (nspui - 1)) * 2)
            x = convolve(x, duob_h)[: len(t)]
        self.ideal_signal = x

        # Find the ideal crossing times, for subsequent jitter analysis of transmitted signal.
        ideal_xings = find_crossings(t, x, decision_scaler, mod_type=mod_type)
        self.ideal_xings = ideal_xings

        # Calculate the channel output.
        #
        # Note: We're not using 'self.ideal_signal', because we rely on the system response to
        #       create the duobinary waveform. We only create it explicitly, above,
        #       so that we'll have an ideal reference for comparison.
        split_time = clock()
        chnl_h = self.calc_chnl_h()
        _calc_chnl_time = clock() - split_time
        split_time = clock()
        chnl_out = convolve(self.x, chnl_h)[: len(t)]
        _conv_chnl_time = clock() - split_time
        if self.debug:
            self.log(f"Channel calculation time: {_calc_chnl_time}")
            self.log(f"Channel convolution time: {_conv_chnl_time}")

        self.channel_perf = nbits * nspb / (clock() - start_time)
        split_time = clock()
        self.status = "Running Tx...(sweep %d of %d)" % (sweep_num, num_sweeps)
    except Exception:
        self.status = "Exception: channel"
        raise

    self.chnl_out = chnl_out
    self.chnl_out_H = fft(chnl_out)

    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the Tx.
    try:
        if self.tx_use_ami:
            # Note: Within the PyBERT computational environment, we use normalized impulse responses,
            #       which have units of (V/ts), where 'ts' is the sample interval. However, IBIS-AMI models expect
            #       units of (V/s). So, we have to scale accordingly, as we transit the boundary between these two worlds.
            tx_cfg = self._tx_cfg  # Grab the 'AMIParamConfigurator' instance for this model.
            # Get the model invoked and initialized, except for 'channel_response', which
            # we need to do several different ways, in order to gather all the data we need.
            tx_param_dict = tx_cfg.input_ami_params
            tx_model_init = AMIModelInitializer(tx_param_dict)
            tx_model_init.sample_interval = ts  # Must be set, before 'channel_response'!
            # Start with a delta function, to capture the model's impulse response.
            tx_model_init.channel_response = [1.0 / ts] + [0.0] * (len(chnl_h) - 1)
            tx_model_init.bit_time = ui
            tx_model = AMIModel(self.tx_dll_file)
            tx_model.initialize(tx_model_init)
            self.log(
                "Tx IBIS-AMI model initialization results:\nInput parameters: {}\nOutput parameters: {}\nMessage: {}".format(
                    tx_model.ami_params_in.decode("utf-8"),
                    tx_model.ami_params_out.decode("utf-8"),
                    tx_model.msg.decode("utf-8"),
                )
            )
            if tx_cfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"]):
                tx_h = array(tx_model.initOut) * ts
            elif not tx_cfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]):
                self.status = "Simulation Error."
                self.log(
                    "ERROR: Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!\n \
I cannot continue.\nYou will have to select a different model.",
                    alert=True,
                )
                return
            elif not self.tx_use_getwave:
                self.status = "Simulation Error."
                self.log(
                    "ERROR: You have elected not to use GetWave for a model, which does not return an impulse response!\n \
I cannot continue.\nPlease, select 'Use GetWave' and try again.",
                    alert=True,
                )
                return
            if self.tx_use_getwave:
                try:
                    tx_s = getwave_step_resp(tx_model)
                except RuntimeError as err:
                    self.status = "Tx GetWave() Error."
                    self.log("ERROR: Never saw a rising step come out of Tx GetWave()!", alert=True)
                    return
                tx_h, _ = trim_impulse(diff(tx_s))
                tx_out, _ = tx_model.getWave(self.x)
            else:  # Init()-only.
                tx_out = convolve(tx_h, self.x)
                tx_s = tx_h.cumsum()
            self.tx_model = tx_model
        else:
            # - Generate the ideal, post-preemphasis signal.
            # To consider: use 'scipy.interp()'. This is what Mark does, in order to induce jitter in the Tx output.
            ffe_out = convolve(symbols, ffe)[: len(symbols)]
            if self.use_ch_file:
                self.rel_power = mean(ffe_out**2) / self.rs
            else:
                self.rel_power = mean(ffe_out**2) / self.Z0
            tx_out = repeat(ffe_out, nspui)  # oversampled output

            # - Calculate the responses.
            # - (The Tx is unique in that the calculated responses aren't used to form the output.
            #    This is partly due to the out of order nature in which we combine the Tx and channel,
            #    and partly due to the fact that we're adding noise to the Tx output.)
            tx_h = array(sum([[x] + list(zeros(nspui - 1)) for x in ffe], []))  # Using sum to concatenate.
            tx_h.resize(len(chnl_h), refcheck=False)  # "refcheck=False", to get around Tox failure.
            tx_s = tx_h.cumsum()
        tx_out.resize(len(t))
        temp = tx_h.copy()
        temp.resize(len(t), refcheck=False)
        tx_H = fft(temp)
        tx_H *= tx_s[-1] / abs(tx_H[0])

        # - Generate the uncorrelated periodic noise. (Assume capacitive coupling.)
        #   - Generate the ideal rectangular aggressor waveform.
        pn_period = 1.0 / pn_freq
        pn_samps = int(pn_period / Ts + 0.5)
        pn = zeros(pn_samps)
        pn[pn_samps // 2 :] = pn_mag
        pn = resize(pn, len(tx_out))
        #   - High pass filter it. (Simulating capacitive coupling.)
        (b, a) = iirfilter(2, gFc / (fs / 2), btype="highpass")
        pn = lfilter(b, a, pn)[: len(pn)]

        # - Add the uncorrelated periodic and random noise to the Tx output.
        tx_out += pn
        tx_out += normal(scale=rn, size=(len(tx_out),))

        # - Convolve w/ channel.
        tx_out_h = convolve(tx_h, chnl_h)[: len(chnl_h)]
        temp = tx_out_h.copy()
        temp.resize(len(t), refcheck=False)
        tx_out_H = fft(temp)
        rx_in = convolve(tx_out, chnl_h)[: len(tx_out)]

        self.tx_s = tx_s
        self.tx_out = tx_out
        self.rx_in = rx_in
        self.tx_out_s = tx_out_h.cumsum()
        self.tx_out_p = self.tx_out_s[nspui:] - self.tx_out_s[:-nspui]
        self.tx_H = tx_H
        self.tx_h = tx_h
        self.tx_out_H = tx_out_H
        self.tx_out_h = tx_out_h

        self.tx_perf = nbits * nspb / (clock() - split_time)
        split_time = clock()
        self.status = "Running CTLE...(sweep %d of %d)" % (sweep_num, num_sweeps)
    except Exception:
        self.status = "Exception: Tx"
        raise

    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the CTLE.
    try:
        if self.rx_use_ami:
            rx_cfg = self._rx_cfg  # Grab the 'AMIParamConfigurator' instance for this model.
            # Get the model invoked and initialized, except for 'channel_response', which
            # we need to do several different ways, in order to gather all the data we need.
            rx_param_dict = rx_cfg.input_ami_params
            rx_model_init = AMIModelInitializer(rx_param_dict)
            rx_model_init.sample_interval = ts  # Must be set, before 'channel_response'!
            rx_model_init.channel_response = tx_out_h / ts
            rx_model_init.bit_time = ui
            rx_model = AMIModel(self.rx_dll_file)
            rx_model.initialize(rx_model_init)
            self.log(
                "Rx IBIS-AMI model initialization results:\nInput parameters: {}\nMessage: {}\nOutput parameters: {}".format(
                    rx_model.ami_params_in.decode("utf-8"),
                    rx_model.msg.decode("utf-8"),
                    rx_model.ami_params_out.decode("utf-8"),
                )
            )
            if rx_cfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"]):
                ctle_out_h = array(rx_model.initOut) * ts
            elif not rx_cfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]):
                self.status = "Simulation Error."
                self.log(
                    "ERROR: Both 'Init_Returns_Impulse' and 'GetWave_Exists' are False!\n \
I cannot continue.\nYou will have to select a different model.",
                    alert=True,
                )
                return
            elif not self.rx_use_getwave:
                self.status = "Simulation Error."
                self.log(
                    "ERROR: You have elected not to use GetWave for a model, which does not return an impulse response!\n \
I cannot continue.\nPlease, select 'Use GetWave' and try again.",
                    alert=True,
                )
                return
            if self.rx_use_getwave:
                try:
                    ctle_s = getwave_step_resp(rx_model)
                except RuntimeError as err:
                    self.status = "Rx GetWave() Error."
                    self.log("ERROR: Never saw a rising step come out of Rx GetWave()!", alert=True)
                    return
                ctle_h = diff(ctle_s)
                temp = ctle_h.copy()
                temp.resize(len(t))
                ctle_H = fft(temp)
                ctle_h.resize(len(chnl_h))
                ctle_out_h = convolve(ctle_h, tx_out_h)[: len(chnl_h)]
            else:  # Init() only.
                ctle_out_h_padded = pad(
                    ctle_out_h,
                    (nspb, len(rx_in) - nspb - len(ctle_out_h)),
                    "linear_ramp",
                    end_values=(0.0, 0.0),
                )
                tx_out_h_padded = pad(
                    tx_out_h,
                    (nspb, len(rx_in) - nspb - len(tx_out_h)),
                    "linear_ramp",
                    end_values=(0.0, 0.0),
                )
                ctle_H = fft(ctle_out_h_padded) / fft(tx_out_h_padded)  # ToDo: I think this is wrong.
                ctle_h = irfft(ctle_H)  # I shouldn't be sending the output of `fft()` into `irfft()`, should I?
                ctle_h.resize(len(chnl_h))
            ctle_s = ctle_h.cumsum()
            ctle_out = convolve(rx_in, ctle_h)
        else:
            if self.use_ctle_file:
                # FIXME: The new import_channel() implementation breaks this:
                ctle_h = import_channel(self.ctle_file, ts, self.f)
                if max(abs(ctle_h)) < 100.0:  # step response?
                    ctle_h = diff(ctle_h)  # impulse response is derivative of step response.
                else:
                    ctle_h *= ts  # Normalize to (V/sample)
                ctle_h.resize(len(t))
                ctle_H = fft(ctle_h)
                ctle_H *= sum(ctle_h) / ctle_H[0]
            else:
                _, ctle_H = make_ctle(rx_bw, peak_freq, peak_mag, w, ctle_mode, ctle_offset)
                ctle_h = irfft(ctle_H)
            ctle_h.resize(len(chnl_h), refcheck=False)
            ctle_out = convolve(rx_in, ctle_h)
            ctle_out -= mean(ctle_out)  # Force zero mean.
            if self.ctle_mode == "AGC":  # Automatic gain control engaged?
                ctle_out *= 2.0 * decision_scaler / ctle_out.ptp()
            ctle_s = ctle_h.cumsum()
            ctle_out_h = convolve(tx_out_h, ctle_h)[: len(tx_out_h)]
        ctle_out.resize(len(t), refcheck=False)
        ctle_out_h_main_lobe = where(ctle_out_h >= max(ctle_out_h) / 2.0)[0]
        if ctle_out_h_main_lobe.size:
            conv_dly_ix = ctle_out_h_main_lobe[0]
        else:
            conv_dly_ix = int(self.chnl_dly // Ts)
        conv_dly = t[conv_dly_ix]  # Keep this line only.
        ctle_out_s = ctle_out_h.cumsum()
        temp = ctle_out_h.copy()
        temp.resize(len(t), refcheck=False)
        ctle_out_H = fft(temp)
        # - Store local variables to class instance.
        # Consider changing this; it could be sensitive to insufficient "front porch" in the CTLE output step response.
        self.ctle_out_p = ctle_out_s[nspui:] - ctle_out_s[:-nspui]
        self.ctle_H = ctle_H
        self.ctle_h = ctle_h
        self.ctle_s = ctle_s
        self.ctle_out_H = ctle_out_H
        self.ctle_out_h = ctle_out_h
        self.ctle_out_s = ctle_out_s
        self.ctle_out = ctle_out
        self.conv_dly = conv_dly
        self.conv_dly_ix = conv_dly_ix

        self.ctle_perf = nbits * nspb / (clock() - split_time)
        split_time = clock()
        self.status = "Running DFE/CDR...(sweep %d of %d)" % (sweep_num, num_sweeps)
    except Exception:
        self.status = "Exception: Rx"
        raise

    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the DFE.
    try:
        if self.use_dfe:
            dfe = DFE(
                n_taps,
                gain,
                delta_t,
                alpha,
                ui,
                nspui,
                decision_scaler,
                mod_type,
                n_ave=n_ave,
                n_lock_ave=n_lock_ave,
                rel_lock_tol=rel_lock_tol,
                lock_sustain=lock_sustain,
                bandwidth=bandwidth,
                ideal=self.sum_ideal,
            )
        else:
            dfe = DFE(
                n_taps,
                0.0,
                delta_t,
                alpha,
                ui,
                nspui,
                decision_scaler,
                mod_type,
                n_ave=n_ave,
                n_lock_ave=n_lock_ave,
                rel_lock_tol=rel_lock_tol,
                lock_sustain=lock_sustain,
                bandwidth=bandwidth,
                ideal=True,
            )
        (dfe_out, tap_weights, ui_ests, clocks, lockeds, clock_times, bits_out) = dfe.run(t, ctle_out)
        dfe_out = array(dfe_out)
        dfe_out.resize(len(t))
        bits_out = array(bits_out)
        auto_corr = (
            1.0
            * correlate(bits_out[(nbits - eye_bits) :], bits[(nbits - eye_bits) :], mode="same")
            / sum(bits[(nbits - eye_bits) :])
        )
        auto_corr = auto_corr[len(auto_corr) // 2 :]
        self.auto_corr = auto_corr
        bit_dly = where(auto_corr == max(auto_corr))[0][0]
        bits_ref = bits[(nbits - eye_bits) :]
        bits_tst = bits_out[(nbits + bit_dly - eye_bits) :]
        if len(bits_ref) > len(bits_tst):
            bits_ref = bits_ref[: len(bits_tst)]
        elif len(bits_tst) > len(bits_ref):
            bits_tst = bits_tst[: len(bits_ref)]
        bit_errs = where(bits_tst ^ bits_ref)[0]
        self.bit_errs = len(bit_errs)

        dfe_h = array([1.0] + list(zeros(nspb - 1)) + sum([[-x] + list(zeros(nspb - 1)) for x in tap_weights[-1]], []))
        dfe_h.resize(len(ctle_out_h), refcheck=False)
        temp = dfe_h.copy()
        temp.resize(len(t), refcheck=False)
        dfe_H = fft(temp)
        self.dfe_s = dfe_h.cumsum()
        dfe_out_H = ctle_out_H * dfe_H
        dfe_out_h = convolve(ctle_out_h, dfe_h)[: len(ctle_out_h)]
        dfe_out_s = dfe_out_h.cumsum()
        self.dfe_out_p = dfe_out_s - pad(dfe_out_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))
        self.dfe_H = dfe_H
        self.dfe_h = dfe_h
        self.dfe_out_H = dfe_out_H
        self.dfe_out_h = dfe_out_h
        self.dfe_out_s = dfe_out_s
        self.dfe_out = dfe_out

        self.dfe_perf = nbits * nspb / (clock() - split_time)
        split_time = clock()
        self.status = "Analyzing jitter...(sweep %d of %d)" % (sweep_num, num_sweeps)
    except Exception:
        self.status = "Exception: DFE"
        raise

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
    try:
        pattern_len = (pow(2, max(pattern)) - 1) * 2

        # - channel output
        len_x_m1 = len(x) - 1
        actual_xings = find_crossings(t, chnl_out, decision_scaler, mod_type=mod_type)
        ofst = (argmax(sig.correlate(chnl_out, x)) - len_x_m1) * Ts
        actual_xings -= ofst
        (
            _,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            _,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
        ) = calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)
        self.t_jitter = t_jitter
        self.isi_chnl = isi
        self.dcd_chnl = dcd
        self.pj_chnl = pj
        self.rj_chnl = rj
        self.thresh_chnl = thresh
        self.jitter_chnl = hist
        self.jitter_ext_chnl = hist_synth
        self.jitter_bins = bin_centers
        self.jitter_spectrum_chnl = jitter_spectrum
        self.jitter_ind_spectrum_chnl = jitter_ind_spectrum
        self.f_MHz = array(spectrum_freqs) * 1.0e-6

        # - Tx output
        actual_xings = find_crossings(t, rx_in, decision_scaler, mod_type=mod_type)
        ofst = (argmax(sig.correlate(rx_in, x)) - len_x_m1) * Ts
        actual_xings -= ofst
        (
            _,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            _,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
        ) = calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)
        self.isi_tx = isi
        self.dcd_tx = dcd
        self.pj_tx = pj
        self.rj_tx = rj
        self.thresh_tx = thresh
        self.jitter_tx = hist
        self.jitter_ext_tx = hist_synth
        self.jitter_spectrum_tx = jitter_spectrum
        self.jitter_ind_spectrum_tx = jitter_ind_spectrum

        # - CTLE output
        actual_xings = find_crossings(t, ctle_out, decision_scaler, mod_type=mod_type)
        ofst = (argmax(sig.correlate(ctle_out, x)) - len_x_m1) * Ts
        actual_xings -= ofst
        (
            jitter,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            jitter_ext,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
        ) = calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)
        self.isi_ctle = isi
        self.dcd_ctle = dcd
        self.pj_ctle = pj
        self.rj_ctle = rj
        self.thresh_ctle = thresh
        self.jitter_ctle = hist
        self.jitter_ext_ctle = hist_synth
        self.jitter_spectrum_ctle = jitter_spectrum
        self.jitter_ind_spectrum_ctle = jitter_ind_spectrum

        # - DFE output
        ignore_until = (nui - eye_uis) * ui
        ideal_xings = array(list(filter(lambda x: x >= ignore_until, ideal_xings)))
        ideal_xings -= ignore_until
        actual_xings = find_crossings(t, dfe_out, decision_scaler, mod_type=mod_type)
        ofst = (argmax(sig.correlate(dfe_out, x)) - len_x_m1) * Ts
        actual_xings -= ofst
        actual_xings = array(list(filter(lambda x: x >= ignore_until, actual_xings)))
        actual_xings -= ignore_until
        (
            jitter,
            t_jitter,
            isi,
            dcd,
            pj,
            rj,
            jitter_ext,
            thresh,
            jitter_spectrum,
            jitter_ind_spectrum,
            spectrum_freqs,
            hist,
            hist_synth,
            bin_centers,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings, actual_xings, rel_thresh)
        self.isi_dfe = isi
        self.dcd_dfe = dcd
        self.pj_dfe = pj
        self.rj_dfe = rj
        self.thresh_dfe = thresh
        self.jitter_dfe = hist
        self.jitter_ext_dfe = hist_synth
        self.jitter_spectrum_dfe = jitter_spectrum
        self.jitter_ind_spectrum_dfe = jitter_ind_spectrum
        self.f_MHz_dfe = array(spectrum_freqs) * 1.0e-6
        dfe_spec = self.jitter_spectrum_dfe
        self.jitter_rejection_ratio = zeros(len(dfe_spec))

        self.jitter_perf = nbits * nspb / (clock() - split_time)
        self.total_perf = nbits * nspb / (clock() - start_time)
        split_time = clock()
        self.status = "Updating plots...(sweep %d of %d)" % (sweep_num, num_sweeps)
    except Exception:
        self.status = "Exception: jitter"
        raise

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

    Ts = t[1]
    ignore_until = (num_ui - eye_uis) * ui
    ignore_samps = (num_ui - eye_uis) * samps_per_ui

    # Misc.
    f_GHz = f[: len(f) // 2] / 1.0e9
    len_f_GHz = len(f_GHz)
    len_t = len(t_ns)
    self.plotdata.set_data("f_GHz", f_GHz[1:])
    self.plotdata.set_data("t_ns", t_ns)
    self.plotdata.set_data("t_ns_chnl", t_ns_chnl)

    # DFE.
    tap_weights = transpose(array(self.adaptation))
    i = 1
    for tap_weight in tap_weights:
        self.plotdata.set_data("tap%d_weights" % i, tap_weight)
        i += 1
    self.plotdata.set_data("tap_weight_index", list(range(len(tap_weight))))
    if self._old_n_taps != n_taps:
        new_plot = Plot(
            self.plotdata,
            auto_colors=["red", "orange", "yellow", "green", "blue", "purple"],
            padding_left=75,
        )
        for i in range(self.n_taps):
            new_plot.plot(
                ("tap_weight_index", "tap%d_weights" % (i + 1)),
                type="line",
                color="auto",
                name="tap%d" % (i + 1),
            )
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
    clock_spec = fft(clock_pers[start_ix:])
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
    self.plotdata.set_data("chnl_H", 20.0 * safe_log10(abs(self.chnl_H[1:len_f_GHz])))
    self.plotdata.set_data("chnl_trimmed_H", 20.0 * safe_log10(abs(self.chnl_trimmed_H[1:len_f_GHz])))
    self.plotdata.set_data("tx_H", 20.0 * safe_log10(abs(self.tx_H[1:len_f_GHz])))
    self.plotdata.set_data("tx_out_H", 20.0 * safe_log10(abs(self.tx_out_H[1:len_f_GHz])))
    self.plotdata.set_data("ctle_H", 20.0 * safe_log10(abs(self.ctle_H[1:len_f_GHz])))
    self.plotdata.set_data("ctle_out_H", 20.0 * safe_log10(abs(self.ctle_out_H[1:len_f_GHz])))
    self.plotdata.set_data("dfe_H", 20.0 * safe_log10(abs(self.dfe_H[1:len_f_GHz])))
    self.plotdata.set_data("dfe_out_H", 20.0 * safe_log10(abs(self.dfe_out_H[1:len_f_GHz])))

    # Jitter distributions
    jitter_ext_chnl = self.jitter_ext_chnl  # These are used, again, in bathtub curve generation, below.
    jitter_ext_tx = self.jitter_ext_tx
    jitter_ext_ctle = self.jitter_ext_ctle
    jitter_ext_dfe = self.jitter_ext_dfe
    self.plotdata.set_data("jitter_bins", array(self.jitter_bins) * 1.0e12)
    self.plotdata.set_data("jitter_chnl", self.jitter_chnl)
    self.plotdata.set_data("jitter_ext_chnl", jitter_ext_chnl)
    self.plotdata.set_data("jitter_tx", self.jitter_tx)
    self.plotdata.set_data("jitter_ext_tx", jitter_ext_tx)
    self.plotdata.set_data("jitter_ctle", self.jitter_ctle)
    self.plotdata.set_data("jitter_ext_ctle", jitter_ext_ctle)
    self.plotdata.set_data("jitter_dfe", self.jitter_dfe)
    self.plotdata.set_data("jitter_ext_dfe", jitter_ext_dfe)

    # Jitter spectrums
    log10_ui = safe_log10(ui)
    self.plotdata.set_data("f_MHz", self.f_MHz[1:])
    self.plotdata.set_data("f_MHz_dfe", self.f_MHz_dfe[1:])
    self.plotdata.set_data("jitter_spectrum_chnl", 10.0 * (safe_log10(self.jitter_spectrum_chnl[1:]) - log10_ui))
    self.plotdata.set_data(
        "jitter_ind_spectrum_chnl", 10.0 * (safe_log10(self.jitter_ind_spectrum_chnl[1:]) - log10_ui)
    )
    self.plotdata.set_data("thresh_chnl", 10.0 * (safe_log10(self.thresh_chnl[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_tx", 10.0 * (safe_log10(self.jitter_spectrum_tx[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_tx", 10.0 * (safe_log10(self.jitter_ind_spectrum_tx[1:]) - log10_ui))
    self.plotdata.set_data("thresh_tx", 10.0 * (safe_log10(self.thresh_tx[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_ctle", 10.0 * (safe_log10(self.jitter_spectrum_ctle[1:]) - log10_ui))
    self.plotdata.set_data(
        "jitter_ind_spectrum_ctle", 10.0 * (safe_log10(self.jitter_ind_spectrum_ctle[1:]) - log10_ui)
    )
    self.plotdata.set_data("thresh_ctle", 10.0 * (safe_log10(self.thresh_ctle[1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_dfe", 10.0 * (safe_log10(self.jitter_spectrum_dfe[1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_dfe", 10.0 * (safe_log10(self.jitter_ind_spectrum_dfe[1:]) - log10_ui))
    self.plotdata.set_data("thresh_dfe", 10.0 * (safe_log10(self.thresh_dfe[1:]) - log10_ui))
    self.plotdata.set_data("jitter_rejection_ratio", self.jitter_rejection_ratio[1:])

    # Bathtubs
    half_len = len(jitter_ext_chnl) // 2
    #  - Channel
    bathtub_chnl = list(cumsum(jitter_ext_chnl[-1 : -(half_len + 1) : -1]))
    bathtub_chnl.reverse()
    bathtub_chnl = array(bathtub_chnl + list(cumsum(jitter_ext_chnl[: half_len + 1])))
    bathtub_chnl = where(
        bathtub_chnl < MIN_BATHTUB_VAL,
        0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_chnl)),
        bathtub_chnl,
    )  # To avoid Chaco log scale plot wierdness.
    self.plotdata.set_data("bathtub_chnl", safe_log10(bathtub_chnl))
    #  - Tx
    bathtub_tx = list(cumsum(jitter_ext_tx[-1 : -(half_len + 1) : -1]))
    bathtub_tx.reverse()
    bathtub_tx = array(bathtub_tx + list(cumsum(jitter_ext_tx[: half_len + 1])))
    bathtub_tx = where(
        bathtub_tx < MIN_BATHTUB_VAL, 0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_tx)), bathtub_tx
    )  # To avoid Chaco log scale plot wierdness.
    self.plotdata.set_data("bathtub_tx", safe_log10(bathtub_tx))
    #  - CTLE
    bathtub_ctle = list(cumsum(jitter_ext_ctle[-1 : -(half_len + 1) : -1]))
    bathtub_ctle.reverse()
    bathtub_ctle = array(bathtub_ctle + list(cumsum(jitter_ext_ctle[: half_len + 1])))
    bathtub_ctle = where(
        bathtub_ctle < MIN_BATHTUB_VAL,
        0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_ctle)),
        bathtub_ctle,
    )  # To avoid Chaco log scale plot wierdness.
    self.plotdata.set_data("bathtub_ctle", safe_log10(bathtub_ctle))
    #  - DFE
    bathtub_dfe = list(cumsum(jitter_ext_dfe[-1 : -(half_len + 1) : -1]))
    bathtub_dfe.reverse()
    bathtub_dfe = array(bathtub_dfe + list(cumsum(jitter_ext_dfe[: half_len + 1])))
    bathtub_dfe = where(
        bathtub_dfe < MIN_BATHTUB_VAL, 0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_dfe)), bathtub_dfe
    )  # To avoid Chaco log scale plot wierdness.
    self.plotdata.set_data("bathtub_dfe", safe_log10(bathtub_dfe))

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
