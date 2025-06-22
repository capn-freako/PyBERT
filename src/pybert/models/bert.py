"""Default controller definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

# pylint: disable=too-many-lines

from time import perf_counter
from typing import Any, Callable, Optional, TypeAlias

import numpy        as np
import numpy.typing as npt
import scipy.signal as sig
from numpy import (  # type: ignore
    argmax,
    array,
    concatenate,
    convolve,
    correlate,
    diff,
    float64,
    histogram,
    linspace,
    mean,
    repeat,
    resize,
    transpose,
    where,
    zeros,
)
from numpy.fft import rfft, irfft  # type: ignore
from numpy.random import normal  # type: ignore
from numpy.typing import NDArray  # type: ignore
from scipy.signal import iirfilter, lfilter
from scipy.interpolate import interp1d

from pyibisami.ami.parser import AmiName, AmiNode, ami_parse
from pybert.models.dfe import DFE
from pybert.utility import (
    calc_eye,
    calc_jitter,
    calc_resps,
    find_crossings,
    import_channel,
    make_bathtub,
    make_ctle,
    run_ami_model,
    safe_log10,
    trim_impulse,
)
from pybert.models.viterbi import ViterbiDecoder_ISI

clock = perf_counter

AmiFloats: TypeAlias = tuple[AmiName, list["float | 'AmiFloats'"]]

DEBUG           = False
MIN_BATHTUB_VAL = 1.0e-12
gFc             = 1.0e6  # Corner frequency of high-pass filter used to model capacitive coupling of periodic noise.


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

    start_time = clock()
    self.status = "Running channel..."

    # The user sets `seed` to zero to indicate that she wants new bits generated for each run.
    if not self.seed:
        self.run_count += 1  # Force regeneration of bit stream.

    # Pull class variables into local storage, performing unit conversion where necessary.
    t = self.t
    t_irfft = self.t_irfft
    f = self.f
    w = self.w
    bits = self.bits
    symbols = self.symbols
    ffe = self.ffe
    nbits = self.nbits
    nui = self.nui
    eye_bits = self.eye_bits
    eye_uis = self.eye_uis
    nspui = self.nspui
    rn = self.rn
    pn_mag = self.pn_mag
    pn_freq = self.pn_freq * 1.0e6
    pattern = self.pattern_
    rx_bw = self.rx_bw * 1.0e9
    peak_freq = self.peak_freq * 1.0e9
    peak_mag = self.peak_mag
    ctle_enable = self.ctle_enable
    delta_t = self.delta_t * 1.0e-12
    alpha = self.alpha
    ui = self.ui
    gain = self.gain
    n_ave = self.n_ave
    decision_scaler = self.decision_scaler
    n_lock_ave = self.n_lock_ave
    dfe_tap_tuners = self.dfe_tap_tuners
    rel_lock_tol = self.rel_lock_tol
    lock_sustain = self.lock_sustain
    bandwidth = self.sum_bw * 1.0e9
    rel_thresh = self.thresh
    mod_type = self.mod_type[0]
    impulse_length = self.impulse_length

    # Calculate misc. values.
    Ts = t[1]
    ts = Ts
    fs = 1 / ts
    min_len =  30 * nspui
    max_len = 100 * nspui
    if impulse_length:
        min_len = max_len = round(impulse_length / ts)
    if mod_type == 2:  # PAM-4
        nspb = nspui // 2
    else:
        nspb = nspui

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
    self.status = "Running Tx..."

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
            # FIXME: The new import_channel() implementation breaks this.
            ctle_h = import_channel(self.ctle_file, ts, f)
            if max(abs(ctle_h)) < 100.0:  # step response?
                ctle_h = diff(ctle_h)  # impulse response is derivative of step response.
            else:
                ctle_h *= ts  # Normalize to (V/sample)
            ctle_h.resize(len(t))
            ctle_H = rfft(ctle_h)  # ToDo: This needs interpolation first.
        else:
            if ctle_enable:
                _, ctle_H = make_ctle(rx_bw, peak_freq, peak_mag, w)
                _ctle_h = irfft(ctle_H)
                krnl = interp1d(t_irfft, _ctle_h, bounds_error=False, fill_value=0)
                ctle_h = krnl(t)
                ctle_h *= sum(_ctle_h) / sum(ctle_h)
                ctle_h, _ = trim_impulse(ctle_h, front_porch=False, min_len=min_len, max_len=max_len)
            else:
                ctle_h = array([1.] + [0. for _ in range(min_len - 1)])
        return ctle_h

    ctle_s = None
    clock_times = None
    try:
        params: list[str] = []
        if self.tx_use_ami and self.tx_use_getwave:
            tx_out, _, tx_h, tx_out_h, msg, _params = run_ami_model(
                self.tx_dll_file, self._tx_cfg, True, ui, ts, chnl_h, x)
            params = _params
            self.log(f"Tx IBIS-AMI model initialization results:\n{msg}")
            tx_getwave_params = list(map(ami_parse, params))
            self.log(f"Tx IBIS-AMI model GetWave() output parameters:\n{tx_getwave_params}")
            rx_in = convolve(tx_out + noise, chnl_h)[:len(tx_out)]
            # Calculate the remaining responses from the impulse responses.
            tx_s, tx_p, tx_H = calc_resps(t, tx_h, ui, f)
            tx_out_s, tx_out_p, tx_out_H = calc_resps(t, tx_out_h, ui, f)
            self.tx_perf = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = "Running CTLE..."
            if self.rx_use_ami and self.rx_use_getwave:
                ctle_out, _, ctle_h, ctle_out_h, msg, _params = run_ami_model(
                    self.rx_dll_file, self._rx_cfg, True, ui, ts, tx_out_h, convolve(tx_out, chnl_h))
                params = _params
                self.log(f"Rx IBIS-AMI model initialization results:\n{msg}")
                _rx_getwave_params = list(map(ami_parse, params))
                self.log(f"Rx IBIS-AMI model GetWave() output parameters:\n{_rx_getwave_params}")
            else:  # Rx is either AMI_Init() or PyBERT native.
                if self.rx_use_ami:  # Rx Init()
                    _, _, ctle_h, ctle_out_h, msg, _ = run_ami_model(
                        self.rx_dll_file, self._rx_cfg, False, ui, ts, chnl_h, tx_out)
                    self.log(f"Rx IBIS-AMI model initialization results:\n{msg}")
                    ctle_out = convolve(tx_out, ctle_out_h)[:len(tx_out)]
                else:                # PyBERT native Rx
                    ctle_h = get_ctle_h()
                    ctle_out_h = convolve(ctle_h, tx_out_h)[:len(ctle_h)]
                    ctle_out = convolve(tx_out, convolve(ctle_h, chnl_h))[:len(tx_out)]
        else:  # Tx is either AMI_Init() or PyBERT native.
            if self.tx_use_ami:  # Tx is AMI_Init().
                rx_in, _, tx_h, tx_out_h, msg, _ = run_ami_model(
                    self.tx_dll_file, self._tx_cfg, False, ui, ts, chnl_h, x)
                self.log(f"Tx IBIS-AMI model initialization results:\n{msg}")
                rx_in += noise
            else:                # Tx is PyBERT native.
                # Using `sum` to concatenate:
                tx_h = array(sum([[x] + list(zeros(nspui - 1)) for x in ffe], []))
                tx_h.resize(len(chnl_h), refcheck=False)  # "refcheck=False", to get around Tox failure.
                tx_out_h = convolve(tx_h, chnl_h)[: len(chnl_h)]
                rx_in = convolve(x, tx_out_h)[:len(x)] + noise
            # Calculate the remaining responses from the impulse responses.
            tx_s, tx_p, tx_H = calc_resps(t, tx_h, ui, f)
            tx_out_s, tx_out_p, tx_out_H = calc_resps(t, tx_out_h, ui, f)
            self.tx_perf = nbits * nspb / (clock() - split_time)
            split_time = clock()
            self.status = "Running CTLE..."
            if self.rx_use_ami and self.rx_use_getwave:
                ctle_out, clock_times, ctle_h, ctle_out_h, msg, _params = run_ami_model(
                    self.rx_dll_file, self._rx_cfg, True, ui, ts, tx_out_h, rx_in)
                params = _params
                self.log(f"Rx IBIS-AMI model initialization results:\n{msg}")
                # Time evolution of (<root_name>: AmiName, <param_vals>: list[AmiNode]):
                # (i.e. - There can be no `AmiAtom`s in the root tuple's second member.)
                rx_getwave_params: list[tuple[AmiName, list[AmiNode]]] = list(map(ami_parse, params))
                param_vals = {}

                def isnumeric(x):
                    try:
                        _ = float(x)
                        return True
                    except:  # noqa: E722, pylint: disable=bare-except
                        return False

                def get_numeric_values(prefix: AmiName, node: AmiNode) -> dict[AmiName, list[np.float64]]:
                    "Retrieve all numeric values from an AMI node, encoding hierarchy in key names."
                    pname = node[0]
                    vals  = node[1]
                    pname_hier = AmiName(prefix + pname)
                    first_val = vals[0]
                    if isnumeric(first_val):
                        return {pname_hier: list(map(float, vals))}  # type: ignore
                    if type(first_val) == AmiNode:  # noqa: E721, pylint: disable=unidiomatic-typecheck
                        subdicts = list(map(lambda nd: get_numeric_values(pname_hier, nd), vals))  # type: ignore
                        rslt = {}
                        for subdict in subdicts:
                            rslt.update(subdict)
                        return rslt
                    return {}

                for nd in rx_getwave_params[0][1]:
                    param_vals.update(get_numeric_values(AmiName(""), nd))
                for rslt in rx_getwave_params[1:]:
                    for nd in rslt[1]:
                        vals_dict = get_numeric_values(AmiName(""), nd)
                        for pname, pvals in vals_dict.items():
                            param_vals[pname].extend(pvals)

                _tap_weights = []
                dfe_tap_keys: list[AmiName] = list(filter(lambda s: s.tolower().contains("tap"), param_vals.keys()))  # type: ignore
                dfe_tap_keys.sort()
                for dfe_tap_key in dfe_tap_keys:
                    _tap_weights.append(param_vals[dfe_tap_key])
                tap_weights: list[list[float]] = list(array(_tap_weights).transpose())
                if "cdr_locked" in param_vals:
                    _lockeds: npt.NDArray[np.float64] = array(param_vals[AmiName("cdr_locked")])
                    _lockeds = _lockeds.repeat(len(t) // len(_lockeds))
                    _lockeds.resize(len(t))
                else:
                    _lockeds = zeros(len(t))
                lockeds: list[bool] = list(map(bool, _lockeds))
                if "cdr_ui" in param_vals:
                    ui_ests: npt.NDArray[np.float64] = array(param_vals[AmiName("cdr_ui")])
                    ui_ests = ui_ests.repeat(len(t) // len(ui_ests))
                    ui_ests.resize(len(t))
                else:
                    ui_ests = zeros(len(t))
            else:  # Rx is either AMI_Init() or PyBERT native.
                if self.rx_use_ami:  # Rx Init()
                    ctle_out, _, ctle_h, ctle_out_h, msg, _ = run_ami_model(
                        self.rx_dll_file, self._rx_cfg, False, ui, ts, tx_out_h, x)
                    self.log(f"Rx IBIS-AMI model initialization results:\n{msg}")
                    ctle_out += noise
                else:                # PyBERT native Rx
                    if ctle_enable:
                        ctle_h = get_ctle_h()
                        ctle_out_h = convolve(tx_out_h, ctle_h)[:len(tx_out_h)]
                        ctle_out = convolve(x + noise, ctle_out_h)[:len(x)]
                    else:
                        ctle_h = array([1.] + [0.] * (min_len - 1))
                        ctle_out_h = tx_out_h
                        ctle_out = rx_in
    except Exception as err:
        self.status = f"Exception: {err}"
        raise

    # Calculate the remaining responses from the impulse responses.
    if ctle_s is None:
        ctle_s, ctle_p, ctle_H = calc_resps(t, ctle_h, ui, f)
    else:
        _, ctle_p, ctle_H = calc_resps(t, ctle_h, ui, f)
    ctle_out_s, ctle_out_p, ctle_out_H = calc_resps(t, ctle_out_h, ui, f)

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
    self.status = "Running DFE/CDR..."

    _check_sim_status()

    # DFE output and incremental/cumulative responses.
    if any(tap.enabled for tap in dfe_tap_tuners):
        _gain = gain
        _ideal = self.sum_ideal
        _n_taps = len(dfe_tap_tuners)
    else:
        _gain = 0.0
        _ideal = True
        _n_taps = 0
    limits = []
    for tuner in self.dfe_tap_tuners:
        if tuner.enabled:
            limits.append((tuner.min_val, tuner.max_val))
        else:
            limits.append((0., 0.))
    dfe = DFE(_n_taps, _gain, delta_t, alpha, ui, nspui, decision_scaler, mod_type,
              n_ave=n_ave, n_lock_ave=n_lock_ave, rel_lock_tol=rel_lock_tol,
              lock_sustain=lock_sustain, bandwidth=bandwidth, ideal=_ideal,
              limits=limits)
    if not (self.rx_use_ami and self.rx_use_getwave):  # Use PyBERT native DFE/CDR.
        dbg_dict: dict[str, Any] = {}
        (dfe_out,
         tap_weights,
         ui_ests,
         clocks,
         lockeds,
         sample_times,
         bits_out) = dfe.run(t, ctle_out, use_agc=self.use_agc, dbg_dict=dbg_dict)
        self.decision_scaler = dfe.decision_scaler
        self.dfe_scalar_values = dbg_dict["scalar_values"]
    else:                                              # Process Rx IBIS-AMI GetWave() output.
        # Process any valid clock times returned by Rx IBIS-AMI model's GetWave() function if apropos.
        dfe_out = array(ctle_out)  # In this case, `ctle_out` includes the effects of IBIS-AMI DFE.
        dfe_out.resize(len(t))
        t_ix = 0
        _bits_out = []
        clocks = zeros(len(t))
        sample_times = []
        if self.rx_use_clocks and clock_times is not None:
            for clock_time in clock_times:
                if clock_time == -1:  # "-1" is used to flag "no more valid clock times".
                    break
                sample_time = clock_time + ui / 2  # IBIS-AMI clock times are edge aligned.
                while t_ix < len(t) and t[t_ix] < sample_time:
                    t_ix += 1
                if t_ix >= len(t):
                    self.log("Went beyond system time vector end searching for next clock time!")
                    break
                _, _bits = dfe.decide(ctle_out[t_ix])
                _bits_out.extend(_bits)
                clocks[t_ix] = 1
                sample_times.append(sample_time)
        # Process any remaining output, using inferred sampling instants.
        if t_ix < (len(t) - 5 * nspui / 4):
            # Starting at `nspui/4` handles either case:
            #   - starting at UI boundary, or
            #   - starting at last sampling instant.
            next_sample_ix = t_ix + nspui // 4 + argmax([sum(abs(ctle_out[t_ix + nspui // 4 + k::nspui]))
                                                         for k in range(nspui)])
            for t_ix in range(next_sample_ix, len(t), nspui):
                _, _bits = dfe.decide(ctle_out[t_ix])
                _bits_out.extend(_bits)
                clocks[t_ix] = 1
                sample_times.append(t[t_ix])
        bits_out = array(_bits_out)
    start_ix = max(0, len(bits_out) - eye_bits)
    end_ix = len(bits_out)
    auto_corr = (
        1.0 * correlate(bits_out[start_ix: end_ix], bits[start_ix: end_ix], mode="same") /  # noqa: W504
        sum(bits[start_ix: end_ix])
    )
    auto_corr = auto_corr[len(auto_corr) // 2:]
    self.auto_corr = auto_corr
    bit_dly = where(auto_corr == max(auto_corr))[0][0]
    first_ref_bit = nbits - eye_bits
    bits_ref = bits[first_ref_bit:]
    first_tst_bit = first_ref_bit + bit_dly
    bits_tst = bits_out[first_tst_bit:]
    if len(bits_ref) > len(bits_tst):
        bits_ref = bits_ref[: len(bits_tst)]
    elif len(bits_tst) > len(bits_ref):
        bits_tst = bits_tst[: len(bits_ref)]
    bit_errs = where(bits_tst ^ bits_ref)[0]
    n_errs = len(bit_errs)
    if n_errs and False:  # pylint: disable=condition-evals-to-constant
        self.log(f"pybert.models.bert.my_run_simulation(): Bit errors detected at indices: {bit_errs}.")
    self.bit_errs = n_errs

    if len(tap_weights) > 0:
        dfe_h = array(
            [1.0] + list(zeros(nspui - 1)) +  # noqa: W504
            sum([[-x] + list(zeros(nspui - 1)) for x in tap_weights[-1]], []))  # sum as concat
        dfe_h.resize(len(ctle_out_h), refcheck=False)
    else:
        dfe_h = array([1.0] + list(zeros(nspui - 1)))
    dfe_out_h = convolve(ctle_out_h, dfe_h)[: len(ctle_out_h)]

    # Calculate the remaining responses from the impulse responses.
    dfe_s, dfe_p, dfe_H = calc_resps(t, dfe_h, ui, f)
    dfe_out_s, dfe_out_p, dfe_out_H = calc_resps(t, dfe_out_h, ui, f)

    self.dfe_perf = nbits * nspb / (clock() - split_time)
    split_time = clock()

    _check_sim_status()

    # Apply Viterbi decoder if apropos.
    self.bit_errs_viterbi = -1
    self.viterbi_perf = 0
    if self.rx_use_viterbi:
        self.status = "Running Viterbi..."
        match mod_type:
            case 0:
                L = 2
            case 1:
                L = 3
            case 2:
                L = 4
            case _:
                raise ValueError(f"Unrecognized modulation type: {mod_type}!")
        N = self.rx_viterbi_symbols
        sigma = 10e-3  # ToDo: Make this an accurate assessment of the random vertical noise.
        dfe_out_p_curs_ix = np.argmax(ctle_out_p)
        dfe_out_p_samps = np.array([ctle_out_p[dfe_out_p_curs_ix + n * nspui] for n in range(N)])
        decoder = ViterbiDecoder_ISI(L, N, sigma, dfe_out_p_samps)
        pulse_resp_samps = []
        for sample_time in filter(lambda x: x <= t[-1], sample_times[first_tst_bit:]):
            ix = np.where(t >= sample_time)[0][0]
            pulse_resp_samps.append(dfe_out[ix])
        if self.debug:
            self.dbg_dict_viterbi = {}
            path = decoder.decode(pulse_resp_samps, dbg_dict=self.dbg_dict_viterbi)
        else:
            path = decoder.decode(pulse_resp_samps)
        symbols_viterbi = list(map(lambda ix: decoder.states[ix][0][-1], path))
        if self.debug:
            self.pulse_resp_samps = pulse_resp_samps
            self.symbols_viterbi = symbols_viterbi
            self.dbg_dict_viterbi["decoder"] = decoder
            self.dbg_dict_viterbi["path"] = path
        bits_out_viterbi = concatenate(list(map(lambda ss: dfe.decide(ss)[1], symbols_viterbi)))
        bits_tst_viterbi = bits_out_viterbi  # [first_tst_bit:]
        if len(bits_ref) > len(bits_tst_viterbi):
            bits_ref = bits_ref[: len(bits_tst_viterbi)]
        elif len(bits_tst_viterbi) > len(bits_ref):
            bits_tst_viterbi = bits_tst_viterbi[: len(bits_ref)]
        num_viterbi_bits = len(bits_tst_viterbi)
        bit_errs_viterbi = where(bits_tst_viterbi ^ bits_ref)[0]
        n_errs_viterbi = len(bit_errs_viterbi)
        self.bit_errs_viterbi = n_errs_viterbi
        self.viterbi_errs_ixs = bit_errs_viterbi
        self.viterbi_perf = num_viterbi_bits * nspb / (clock() - split_time)
        split_time = clock()

    self.dfe_h = dfe_h
    self.dfe_s = dfe_s
    self.dfe_p = dfe_p
    self.dfe_H = dfe_H
    self.dfe_out_h = dfe_out_h
    self.dfe_out_s = dfe_out_s
    self.dfe_out_p = dfe_out_p
    self.dfe_out_H = dfe_out_H
    self.dfe_out = dfe_out
    self.lockeds = lockeds

    self.status = "Analyzing jitter..."

    _check_sim_status()

    # Save local variables to class instance for state preservation, performing unit conversion where necessary.
    self.adaptation = tap_weights
    self.ui_ests = array(ui_ests) * 1.0e12  # (ps)
    self.clocks = clocks
    self.clock_times = sample_times

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

    def eye_xings(xings, ofst=0) -> NDArray[float64]:
        """
        Return crossings from that portion of the signal used to generate the eye.

        Args:
            xings([float]): List of crossings.

        Keyword Args:
            ofst(float): Time offset to be subtracted from all crossings.

        Returns:
            [float]: Selected crossings, offset and eye-start corrected.
        """
        _xings = array(xings) - ofst
        return _xings[where(_xings > xing_min_t)] - xing_min_t

    jit_chnl_done: bool = False
    jit_tx_done: bool = False
    jit_ctle_done: bool = False
    jit_dfe_done: bool = False
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
            mu_pos,
            mu_neg,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.t_jitter  = t_jitter
        self.isi_chnl  = isi
        self.dcd_chnl  = dcd
        self.pj_chnl   = pj
        self.rj_chnl   = rj
        self.pjDD_chnl = pjDD
        self.rjDD_chnl = rjDD
        self.mu_pos_chnl = mu_pos
        self.mu_neg_chnl = mu_neg
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
        jit_chnl_done = True

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
            mu_pos,
            mu_neg,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.isi_tx  = isi
        self.dcd_tx  = dcd
        self.pj_tx   = pj
        self.rj_tx   = rj
        self.pjDD_tx = pjDD
        self.rjDD_tx = rjDD
        self.mu_pos_tx = mu_pos
        self.mu_neg_tx = mu_neg
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
        jit_tx_done = True

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
            mu_pos,
            mu_neg,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh)
        self.isi_ctle  = isi
        self.dcd_ctle  = dcd
        self.pj_ctle   = pj
        self.rj_ctle   = rj
        self.pjDD_ctle = pjDD
        self.rjDD_ctle = rjDD
        self.mu_pos_ctle = mu_pos
        self.mu_neg_ctle = mu_neg
        self.thresh_ctle = thresh
        self.jitter_ctle = hist
        self.jitter_ext_ctle = hist_synth
        self.jitter_spectrum_ctle = jitter_spectrum
        self.jitter_ind_spectrum_ctle = jitter_ind_spectrum
        self.tie_ctle = tie
        self.tie_ind_ctle = tie_ind
        jit_ctle_done = True

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
            mu_pos,
            mu_neg,
        ) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings_jit, actual_xings_jit, rel_thresh, dbg_obj=self)
        self.isi_dfe  = isi
        self.dcd_dfe  = dcd
        self.pj_dfe   = pj
        self.rj_dfe   = rj
        self.pjDD_dfe = pjDD
        self.rjDD_dfe = rjDD
        self.mu_pos_dfe = mu_pos
        self.mu_neg_dfe = mu_neg
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
        jit_dfe_done = True

        self.jitter_perf = nbits * nspb / (clock() - split_time)
        self.total_perf = nbits * nspb / (clock() - start_time)
        split_time = clock()
        self.status = "Updating plots..."
    except ValueError as err:
        self.log(f"The jitter calculation could not be completed, due to the following error:\n{err}",
                 alert=False)
        if jit_chnl_done:
            if jit_tx_done:
                if jit_ctle_done:
                    if jit_dfe_done:
                        self.status = "Exception: But, all finished!"
                    else:
                        self.status = "Exception: DFE jitter"
                else:
                    self.status = "Exception: CTLE jitter"
            else:
                self.status = "Exception: Tx jitter"
        else:
            self.status = "Exception: channel jitter"
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
    except Exception as err:  # pylint: disable=broad-exception-caught
        self.log(f"The following error occured, while trying to update the plots:\n{err}")
        self.status = "Exception: plotting"
        # raise


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
    ui_ests = self.ui_ests
    f = self.f
    t = self.t
    t_ns = self.t_ns
    t_ns_chnl = self.t_ns_chnl
    t_irfft = self.t_irfft

    ignore_until = (num_ui - eye_uis) * ui
    ignore_samps = (num_ui - eye_uis) * samps_per_ui

    # Misc.
    f_GHz = f / 1.0e9
    len_f_GHz = len(f_GHz)
    len_t = len(t_ns)
    self.plotdata.set_data("f_GHz", f_GHz[1:])
    self.plotdata.set_data("t_ns_chnl", t_ns_chnl)
    self.plotdata.set_data("t_ns_irfft", t_irfft * 1e9)
    if len_t > 1000:  # to prevent Chaco plotting error with too much data
        t_ns_plot = linspace(0, t_ns[-1], 1000)
    else:
        t_ns_plot = t_ns
    self.plotdata.set_data("t_ns", t_ns_plot)

    # DFE.
    tap_weights = transpose(array(self.adaptation))
    if len(tap_weights):
        for k, tap_weight in enumerate(tap_weights):  # pylint: disable=undefined-loop-variable
            self.plotdata.set_data(f"tap{k + 1}_weights", tap_weight)
        self.plotdata.set_data("tap_weight_index", list(range(len(tap_weight))))  # pylint: disable=undefined-loop-variable
    else:
        for k in range(len(self.dfe_tap_tuners)):
            self.plotdata.set_data(f"tap{k + 1}_weights", zeros(10))
        self.plotdata.set_data("tap_weight_index", list(range(10)))  # pylint: disable=undefined-loop-variable

    (bin_counts, bin_edges) = histogram(ui_ests, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    clock_spec = rfft(ui_ests)
    _f0 = 1 / (t[1] * len(t))
    spec_freqs = [_f0 * k for k in range(len(t) // 2 + 1)]
    self.plotdata.set_data("clk_per_hist_bins", bin_centers)
    self.plotdata.set_data("clk_per_hist_vals", bin_counts)
    self.plotdata.set_data("clk_spec", safe_log10(abs(clock_spec[1:]) / abs(clock_spec[1])))  # Omit the d.c. value and normalize to fundamental magnitude.
    self.plotdata.set_data("clk_freqs", array(spec_freqs[1:]) * ui)
    self.plotdata.set_data("dfe_out", self.dfe_out)
    self.plotdata.set_data("clocks", self.clocks)
    self.plotdata.set_data("lockeds", self.lockeds)
    if len_t > 1000:  # to prevent Chaco plotting error with too much data
        krnl = interp1d(t_ns, self.ui_ests)
        ui_ests_plot = krnl(t_ns_plot)
    else:
        ui_ests_plot = self.ui_ests
    self.plotdata.set_data("ui_ests", ui_ests_plot)

    # Impulse responses
    self.plotdata.set_data("chnl_h", self.chnl_h)
    self.plotdata.set_data("tx_h", self.tx_h)
    self.plotdata.set_data("tx_out_h", self.tx_out_h)
    self.plotdata.set_data("ctle_h", self.ctle_h)
    self.plotdata.set_data("ctle_out_h", self.ctle_out_h)
    self.plotdata.set_data("dfe_h", self.dfe_h)
    self.plotdata.set_data("dfe_out_h", self.dfe_out_h)

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

    # Outputs
    ideal_signal = self.ideal_signal[:len_t]
    chnl_out = self.chnl_out[:len_t]
    rx_in = self.rx_in[:len_t]
    ctle_out = self.ctle_out[:len_t]
    dfe_out = self.dfe_out[:len_t]
    lockeds = self.lockeds[:len_t]
    if len_t > 1000:  # to prevent Chaco plotting error with too much data
        krnl = interp1d(t_ns, ideal_signal)
        ideal_signal_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, chnl_out)
        chnl_out_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, rx_in)
        rx_in_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, ctle_out)
        ctle_out_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, dfe_out)
        dfe_out_plot = krnl(t_ns_plot)
        krnl = interp1d(t_ns, lockeds)
        lockeds_plot = krnl(t_ns_plot)
    else:
        ideal_signal_plot = ideal_signal
        chnl_out_plot = chnl_out
        rx_in_plot = rx_in
        ctle_out_plot = ctle_out
        dfe_out_plot = dfe_out
        lockeds_plot = lockeds
    self.plotdata.set_data("ideal_signal", ideal_signal_plot)
    self.plotdata.set_data("chnl_out", chnl_out_plot)
    self.plotdata.set_data("rx_in", rx_in_plot)
    self.plotdata.set_data("ctle_out", ctle_out_plot)
    self.plotdata.set_data("dfe_out", dfe_out_plot)
    self.plotdata.set_data("dbg_out", lockeds_plot)

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
    bathtub_chnl = make_bathtub(
        jitter_bins, jitter_chnl, min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_chnl, mu_r=self.mu_pos_chnl, mu_l=self.mu_neg_chnl, extrap=True)
    bathtub_tx = make_bathtub(
        jitter_bins, jitter_tx,   min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_tx, mu_r=self.mu_pos_tx, mu_l=self.mu_neg_tx, extrap=True)
    bathtub_ctle = make_bathtub(
        jitter_bins, jitter_ctle, min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_ctle, mu_r=self.mu_pos_ctle, mu_l=self.mu_neg_ctle, extrap=True)
    bathtub_dfe = make_bathtub(
        jitter_bins, jitter_dfe,  min_val=0.1 * MIN_BATHTUB_VAL,
        rj=self.rjDD_dfe, mu_r=self.mu_pos_dfe, mu_l=self.mu_neg_dfe, extrap=True)
    self.plotdata.set_data("bathtub_chnl", safe_log10(bathtub_chnl))
    self.plotdata.set_data("bathtub_tx",   safe_log10(bathtub_tx))
    self.plotdata.set_data("bathtub_ctle", safe_log10(bathtub_ctle))
    self.plotdata.set_data("bathtub_dfe",  safe_log10(bathtub_dfe))

    # Eyes
    width = 2 * samps_per_ui
    xs = linspace(-ui * 1.0e12, ui * 1.0e12, width)
    height = 1000
    tiny_noise = normal(scale=1e-3, size=len(chnl_out[ignore_samps:]))  # to make channel eye easier to view.
    chnl_out_noisy = self.chnl_out[ignore_samps:] + tiny_noise
    y_max = 1.1 * max(abs(array(chnl_out_noisy)))
    eye_chnl = calc_eye(ui, samps_per_ui, height, chnl_out_noisy, y_max)
    y_max = 1.1 * max(abs(array(self.rx_in[ignore_samps:])))
    eye_tx = calc_eye(ui, samps_per_ui, height, self.rx_in[ignore_samps:], y_max)
    y_max = 1.1 * max(abs(array(self.ctle_out[ignore_samps:])))
    eye_ctle = calc_eye(ui, samps_per_ui, height, self.ctle_out[ignore_samps:], y_max)
    y_max = 1.1 * max(abs(array(self.dfe_out[ignore_samps:])))
    self.dfe_eye_ymax = y_max
    i = 0
    len_clock_times = len(clock_times)
    while i < len_clock_times and clock_times[i] < ignore_until:
        i += 1
    if i >= len(clock_times):
        self.log("ERROR: Insufficient coverage in 'clock_times' vector.")
        eye_dfe = calc_eye(ui, samps_per_ui, height, self.dfe_out[ignore_samps:], y_max)
    else:
        eye_dfe = calc_eye(ui, samps_per_ui, height, self.dfe_out[ignore_samps:], y_max, array(clock_times[i:]) - ignore_until)
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
