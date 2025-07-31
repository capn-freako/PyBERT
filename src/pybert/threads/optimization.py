"""
PyBERT Linear Equalization Optimizer

Original author: David Banas <capn.freako@gmail.com>

Original date: June 21, 2017

Copyright (c) 2017 David Banas; all rights reserved World wide.

TX, RX or co optimization are run in a separate thread to keep the gui responsive.
"""

import time

from numpy import arange, array, convolve, log10, ones, pi, prod, resize, where, zeros  # type: ignore
from numpy.fft import irfft, rfft  # type: ignore
from scipy.interpolate import interp1d

from pychopmarg.optimize import mmse
from pychopmarg.noise import NoiseCalc

from pybert.models.tx_tap import TxTapTuner
from pybert.threads.stoppable import StoppableThread
from pybert.utility import make_ctle, calc_resps, add_ffe_dfe, get_dfe_weights

gDebugOptimize = False


# pylint: disable=no-member
class OptThread(StoppableThread):
    "Used to run EQ optimization in its own thread, to preserve GUI responsiveness."

    def run(self):
        "Run the equalization optimization thread."

        pybert = self.pybert

        pybert.status = "Optimizing EQ..."
        time.sleep(0.001)

        try:
            tx_weights, rx_peaking, rx_weights, fom, valid = coopt(pybert)
        except RuntimeError as err:
            pybert.log(f"{err}")
            pybert.status = "User abort."
            return

        if not valid:
            pybert.status = "Failed."
            return
        for k, tx_weight in enumerate(tx_weights):
            pybert.tx_tap_tuners[k].value = tx_weight
        pybert.peak_mag_tune = rx_peaking
        for k, rx_weight in enumerate(rx_weights):
            pybert.ffe_tap_tuners[k].value = rx_weight
        pybert.status = f"Finished. (SNR: {20 * log10(fom):5.1f} dB)"


def mk_tx_weights(weightss: list[list[float]], enumerated_tuners: list[tuple[int, TxTapTuner]]) -> list[list[float]]:
    """
    Make all tap weight combinations possible from a list of Tx tap tuners.

    Args:
        weightss: The current list of tap weight combinations. (Supports recursion.)
        enumerated_tuners: List of pairs, each containing

            - the index of this tap in the list, and
            - this tap tuner.

    Return:
        List of all possible tap weight combinations.

    Raises:
        ValueError: If total number of combinations is too large.
    """

    # Check total number of combinations.
    n_combs = prod([int((tuner.max_val - tuner.min_val) / tuner.step + 1)
                    if tuner.enabled else 1 for _, tuner in enumerated_tuners])
    if n_combs > 1_000_000:
        raise ValueError(
            f"Total number of combinations ({int(n_combs // 1e6)} M) is too great!")

    # Check for end of recursion.
    if not enumerated_tuners:
        return weightss

    # Perform normal (i.e. - recursive) calculation.
    head, *tail = enumerated_tuners
    n, tuner = head
    if not tuner.enabled:
        return mk_tx_weights(weightss, tail)
    weight_vals = arange(tuner.min_val, tuner.max_val + tuner.step, tuner.step)
    new_weightss = []
    for weights in weightss:
        for val in weight_vals:
            weights[n] = val
            new_weightss.append(weights.copy())
    return mk_tx_weights(new_weightss, tail)


def coopt(pybert) -> tuple[list[float], float, list[float], float, bool]:  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """
    Co-optimize the Tx/Rx linear equalization, assuming ideal bounded DFE.

    Args:
        pybert(PyBERT): The PyBERT instance on which to perform co-optimization.

    Returns:
        A tuple containing

            - the optimum Tx FFE tap weights,
            - the optimum Rx CTLE peaking,
            - the optimum Rx FFE tap weights,
            - the figure of merit for the returned settings, and
            - the status of the optimization attempt (`True` = success).

    Raises:
        RuntimeError: If user opts to abort.
    """

    # Grab needed quantities from PyBERT instance.
    min_mag   = pybert.min_mag_tune
    max_mag   = pybert.max_mag_tune
    step_mag  = pybert.step_mag_tune
    rx_bw     = pybert.rx_bw_tune * 1e9
    peak_freq = pybert.peak_freq_tune * 1e9
    dfe_taps  = pybert.dfe_tap_tuners
    tx_taps   = pybert.tx_tap_tuners
    rx_taps   = pybert.ffe_tap_tuners
    rx_n_taps = pybert.rx_n_taps
    rx_n_pre  = pybert.rx_n_pre
    max_len   = 100 * pybert.nspui
    num_levels = pybert.mod_type[0] + 2

    # Find number of enabled DFE taps. (No support for floating taps, yet.)
    n_dfe_taps = 0
    for tap in dfe_taps:
        if not tap.enabled:
            break
        n_dfe_taps += 1

    # Calculate time/frequency vectors for CTLE.
    ctle_fmax = 100 * rx_bw  # Should give -40dB at truncation, assuming 20 dB/dec. roll-off.
    f_ctle = arange(0, ctle_fmax + 10e6, 10e6)  # 10 MHz freq. step & includes `ctle_fmax` (i.e. - fNyquist)
    w_ctle = 2 * pi * f_ctle
    ts_ctle = 0.5 / ctle_fmax
    t_ctle = [n * ts_ctle for n in range(2 * (len(f_ctle) - 1))]  # Presumes use of `rfft()`/`irfft()`.

    # Calculate unequalized channel pulse response.
    h_chnl = pybert.calc_chnl_h()
    t = pybert.t
    ui = pybert.ui
    nspui = pybert.nspui
    f = pybert.f
    _, p_chnl, _ = calc_resps(t, h_chnl, ui, f)
    pybert.plotdata.set_data("p_chnl", p_chnl)

    # Calculate Tx tap weight candidates.
    n_weights = len(tx_taps)
    n_enabled_weights = len(list(filter(lambda t: t.enabled, tx_taps)))
    tx_curs_pos = max(0, -tx_taps[0].pos)  # list position at which to insert main tap
    try:
        tx_weightss = mk_tx_weights([[0] * n_weights,], list(enumerate(pybert.tx_tap_tuners)))
    except ValueError as err:
        raise RuntimeError(
            "\n".join([
                f"{err}",
                "Sorry, that's more Tx tap weight combinations than I can handle.",
                "I had to abort the EQ optimization in your stead.",
            ])) from err

    for tx_weights in tx_weightss:
        tx_weights.insert(tx_curs_pos, 1 - sum(abs(array(tx_weights))))

    # Calculate CTLE gain candidates.
    if pybert.ctle_enable_tune:
        peak_mags = arange(min_mag, max_mag + step_mag, step_mag)
    else:
        peak_mags = array([0])

    # Calculate Rx FFE tap weight candidates.
    n_rx_weights = len(rx_taps)
    n_enabled_rx_weights = len(list(filter(lambda t: t.enabled, rx_taps)))
    if pybert.use_mmse:
        rx_weightss = [[0.0] * n_rx_weights,]
    else:
        try:
            rx_weightss = mk_tx_weights([[0.0] * n_rx_weights,], list(enumerate(rx_taps)))
        except ValueError as err:
            raise RuntimeError(
                "\n".join([
                    f"{err}",
                    "Sorry, that's more Rx FFE tap weight combinations than I can handle.",
                    "I had to abort the EQ optimization in your stead.",
                ])) from err

    # Calculate and report the total number of trials, as well as some other misc. info.
    if pybert.use_mmse:
        n_trials = len(peak_mags) * len(tx_weightss)
    else:
        n_trials = len(peak_mags) * len(tx_weightss) * len(rx_weightss)
    trials_run_inc = n_trials // 100 or 1

    pybert.log("\n".join([
        "Optimizing linear EQ...",
        f"\tTime step: {t[1] * 1e12:5.1f} ps",
        f"\tUnit interval: {ui * 1e12:5.1f} ps",
        f"\tOversampling factor: {nspui}",
        f"\tNumber of enabled Tx taps: {n_enabled_weights}",
        f"\tTx cursor tap position: {tx_curs_pos}",
        f"\tNumber of enabled Rx FFE taps: {n_enabled_rx_weights}",
        f"\tRunning {n_trials} trials.",
        ""]))

    # Calculate `f_t` and interpolated channel frequency response.
    dt = t[1] - t[0]            # `t` assumed uniformly sampled throughout.
    fN = 0.5 / dt               # Nyquist frequency
    f0 = 100e6                  # fundamental frequency
    f_t = arange(0, fN + f0 / 2, f0)  # "+ f0 / 2", to ensure `fN` gets included.
    _t = array([n * dt for n in range((len(f_t) - 1) * 2)])
    krnl = interp1d(f, pybert.chnl_H, bounds_error=False, fill_value=0)
    chnl_H = krnl(f_t)

    # Run the optimization loop.
    fom_max = -1000.
    peak_mag_best = 0.
    trials_run = 0
    dfe_weights = zeros(len(dfe_taps))
    rx_weights_best = zeros(rx_n_taps)
    dfe_weights_best = zeros(len(dfe_taps))
    tx_weights_best = [0.0] * len(tx_taps)
    del tx_weights_best[tx_curs_pos]
    for peak_mag in peak_mags:  # pylint: disable=too-many-nested-blocks
        _, H_ctle = make_ctle(rx_bw, peak_freq, peak_mag, w_ctle)
        _h_ctle = irfft(H_ctle)
        krnl = interp1d(t_ctle, _h_ctle, bounds_error=False, fill_value=0)
        h_ctle = krnl(t[:max_len])
        h_ctle *= sum(_h_ctle) / sum(h_ctle)
        p_ctle_out = convolve(p_chnl, h_ctle)[:len(p_chnl)]
        ctle_H = rfft(resize(h_ctle, len(_t)))
        for tx_weights in tx_weightss:
            # sum = concatenate
            h_tx = array(sum([[tx_weight] + [0] * (nspui - 1) for tx_weight in tx_weights], []))
            p_tx = convolve(p_ctle_out, h_tx)
            p_tx.resize(len(_t), refcheck=False)  # `p_tx = numpy.resize(p_tx, ...)` does NOT work!
            if pybert.use_mmse:
                curs_ix = where(p_tx == max(p_tx))[0][0]
                curs_amp = p_tx[curs_ix]
                n_pre_isi = curs_ix // nspui
                tx_H = rfft(resize(h_tx, len(_t)))
                noise_calc = NoiseCalc(
                    num_levels, ui, curs_ix, _t, p_tx, [], f_t,
                    tx_H, chnl_H, ones(len(f_t)), ctle_H,
                    0.0, 0.5, 25, 0.0, 0.0
                )
                mmse_rslts = mmse(
                    noise_calc, rx_n_taps, rx_n_pre, n_dfe_taps, pybert.rlm, pybert.mod_type[0] + 2,
                    array(list(map(lambda t: t.min_val, dfe_taps[:n_dfe_taps]))), array(list(map(lambda t: t.max_val, dfe_taps[:n_dfe_taps]))),
                    array(list(map(lambda t: t.min_val, rx_taps[:rx_n_taps]))), array(list(map(lambda t: t.max_val, rx_taps[:rx_n_taps]))))
                rx_weights_better = mmse_rslts["rx_taps"]
                dfe_weights_better = mmse_rslts["dfe_tap_weights"]
                fom = mmse_rslts["fom"]
                p_tot = resize(add_ffe_dfe(rx_weights_better, dfe_weights_better, nspui, p_tx),
                               nspui * (n_rx_weights + 5))
                fom_better = fom
                trials_run += 1
                if not trials_run % trials_run_inc:
                    pybert.status = f"Optimizing EQ...({100 * trials_run // n_trials}%)"
                    time.sleep(0.001)
                    if pybert.opt_thread.stopped():
                        pybert.status = "Optimization aborted by user."
                        raise RuntimeError("Optimization aborted by user.")
            else:  # exhaustive sweep of Rx FFE tap weight combinations
                fom_better = fom_max
                for rx_weights in rx_weightss:
                    try:  # FixMe: The line below is broken.
                        p_tot = add_ffe_dfe(rx_weights, get_dfe_weights(dfe_taps, p_tx, nspui), nspui, p_tx)
                        curs_ix = where(p_tot == max(p_tot))[0][0]
                        curs_amp = p_tot[curs_ix]
                    except ValueError:
                        continue
                    n_pre_isi = curs_ix // nspui
                    isi_sum = sum(abs(p_tot[curs_ix - n_pre_isi * nspui::nspui])) - abs(curs_amp)
                    fom = curs_amp / isi_sum
                    if fom > fom_better:
                        rx_weights_better = rx_weights
                        dfe_weights_better = dfe_weights
                        fom_better = fom
                    trials_run += 1
                    if not trials_run % trials_run_inc:
                        pybert.status = f"Optimizing EQ...({100 * trials_run // n_trials}%)"
                        time.sleep(0.001)
                        if pybert.opt_thread.stopped():
                            pybert.status = "Optimization aborted by user."
                            raise RuntimeError("Optimization aborted by user.")
            if fom_better > fom_max:
                rx_weights_best = rx_weights_better.copy()
                dfe_weights_best = dfe_weights_better.copy()
                tx_weights_best = tx_weights.copy()
                del tx_weights_best[tx_curs_pos]
                peak_mag_best = peak_mag
                curs_ix = where(p_tot == max(p_tot))[0][0]
                curs_amp = p_tot[curs_ix]
                n_pre_isi = curs_ix // nspui
                clocks = 1.1 * curs_amp * ones(len(p_tot))
                clocks[curs_ix - n_pre_isi * nspui::nspui] = 0
                pybert.plotdata.set_data("clocks_tune", clocks)
                pybert.plotdata.set_data("ctle_out_h_tune", p_tot)
                pybert.plotdata.set_data("t_ns_opt", pybert.t_ns[:len(p_tot)])
                pybert.plotdata.set_data("curs_amp", [0, curs_amp])
                curs_time = pybert.t_ns[curs_ix]
                pybert.plotdata.set_data("curs_ix", [curs_time, curs_time])
                fom_max = fom_better
                time.sleep(0.001)

    for k, dfe_weight in enumerate(dfe_weights_best):  # pylint: disable=possibly-used-before-assignment
        dfe_taps[k].value = dfe_weight

    return (tx_weights_best, peak_mag_best, list(rx_weights_best), fom_max, True)  # pylint: disable=possibly-used-before-assignment
