"""
PyBERT Linear Equalization Optimizer

Original author: David Banas <capn.freako@gmail.com>

Original date: June 21, 2017

Copyright (c) 2017 David Banas; all rights reserved World wide.

TX, RX or co optimization are run in a separate thread to keep the gui responsive.
"""

import time

from numpy import arange, argmax, array, convolve, log10, ones, pi, where, zeros  # type: ignore
from numpy.fft import irfft  # type: ignore
from scipy.interpolate import interp1d

from pybert.models.tx_tap import TxTapTuner
from pybert.threads.stoppable import StoppableThread
from pybert.utility import make_ctle, calc_resps

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
            tx_weights, rx_peaking, fom, valid = coopt(pybert)
        except RuntimeError:
            pybert.status = "User abort."
            return

        if not valid:
            pybert.status = "Failed."
            return
        for k, tx_weight in enumerate(tx_weights):
            pybert.tx_tap_tuners[k].value = tx_weight
        pybert.peak_mag_tune = rx_peaking
        pybert.status = f"Finished. (SNR: {20 * log10(fom):5.1f} dB)"


def mk_tx_weights(weightss: list[list[float]], enumerated_tuners: list[tuple[int, TxTapTuner]]) -> list[list[float]]:
    """
    Make all tap weight combinations possible from a list of Tx tap tuners.

    Args:
        weightss: The current list of tap weight combinations.
        enumerated_tuners: List of pairs, each containing:
            - the index of this tap in the list, and
            - this tap tuner.

    Return:
        tap_weights: List of all possible tap weight combinations.
    """
    if not enumerated_tuners:
        return weightss
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


def coopt(pybert) -> tuple[list[float], float, float, bool]:  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """
    Co-optimize the Tx/Rx linear equalization, assuming ideal bounded DFE.

    Args:
        pybert(PyBERT): The PyBERT instance on which to perform co-optimization.

    Returns:
        (tx_weights, ctle_peaking, FOM, success): The ideal Tx FFE / Rx CTLE settings & figure of merit.

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
    max_len   = 100 * pybert.nspui

    # Calculate time/frequency vectors for CTLE.
    ctle_fmax = 100 * rx_bw  # Should give -40dB at truncation, assuming 20 dB/dec. roll-off.
    f_ctle = arange(0, ctle_fmax + 10e6, 10e6)  # 10 MHz freq. step & includes `ctle_fmax` (i.e. - fNyquist)
    w_ctle = 2 * pi * f_ctle
    t_ctle = [n * 1.0 / (2 * ctle_fmax) for n in range(2 * (len(f_ctle) - 1))]

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
    tx_curs_pos = max(0, -tx_taps[0].pos)  # list position at which to insert main tap
    tx_weightss = mk_tx_weights([[0 for _ in range(n_weights)],], list(enumerate(pybert.tx_tap_tuners)))
    for tx_weights in tx_weightss:
        tx_weights.insert(tx_curs_pos, 1 - sum(abs(array(tx_weights))))

    # Calculate CTLE gain candidates.
    if pybert.ctle_enable_tune:
        peak_mags = arange(min_mag, max_mag + step_mag, step_mag)
    else:
        peak_mags = array([0])

    # Calculate and report the total number of trials, as well as some other misc. info.
    n_trials = len(peak_mags) * len(tx_weightss)
    pybert.log("\n".join([
        "Optimizing linear EQ...",
        f"\tTime step: {t[1] * 1e12:5.1f} ps",
        f"\tUnit interval: {ui * 1e12:5.1f} ps",
        f"\tOversampling factor: {nspui}",
        f"\tNumber of Tx taps: {n_weights}",
        f"\tTx cursor tap position: {tx_curs_pos}",
        f"\tRunning {n_trials} trials.",
        ""]))

    # Run the optimization loop.
    fom_max = -1000.
    peak_mag_best = 0.
    trials_run = 0
    dfe_weights = zeros(len(dfe_taps))
    for peak_mag in peak_mags:
        _, H_ctle = make_ctle(rx_bw, peak_freq, peak_mag, w_ctle)
        _h_ctle = irfft(H_ctle)
        krnl = interp1d(t_ctle, _h_ctle, bounds_error=False, fill_value=0)
        h_ctle = krnl(t[:max_len])
        h_ctle *= sum(_h_ctle) / sum(h_ctle)
        p_ctle_out = convolve(p_chnl, h_ctle)[:len(p_chnl)]
        for tx_weights in tx_weightss:
            # sum = concatenate
            h_tx = array(sum([[tx_weight] + [0] * (nspui - 1) for tx_weight in tx_weights], []))
            p_tot = convolve(p_ctle_out, h_tx)[:len(p_ctle_out)]
            curs_ix = where(p_tot == max(p_tot))[0][0]
            # Test for obvious "to ignore" cases.
            if p_tot[argmax(abs(p_tot))] < 0:  # Main peak is negative.
                continue
            if curs_ix > len(p_tot) // 2:      # Main peak occurs in right half of waveform.
                continue
            curs_amp = p_tot[curs_ix]
            for k, tap in filter(lambda k_tap: k_tap[1].enabled, enumerate(dfe_taps)):
                isi = p_tot[curs_ix + (k + 1) * nspui]
                ideal_tap_weight = isi / curs_amp
                actual_tap_weight = max(tap.min_val, min(tap.max_val, ideal_tap_weight))
                dfe_weights[k] = actual_tap_weight
                p_tot[curs_ix + (k + 1) * nspui - nspui // 2:] -= actual_tap_weight * curs_amp
            n_pre_isi = curs_ix // nspui
            isi_sum = sum(abs(p_tot[curs_ix - n_pre_isi * nspui::nspui])) - abs(curs_amp)
            fom = curs_amp / isi_sum
            if fom > fom_max:
                dfe_weights_best = dfe_weights.copy()
                tx_weights_best = tx_weights.copy()
                del tx_weights_best[tx_curs_pos]
                peak_mag_best = peak_mag
                clocks = 1.1 * curs_amp * ones(len(p_tot))
                for ix in range(curs_ix - n_pre_isi * nspui, len(clocks), nspui):
                    clocks[ix] = 0
                pybert.plotdata.set_data("clocks_tune", clocks)
                pybert.plotdata.set_data("ctle_out_h_tune", p_tot)
                pybert.plotdata.set_data("t_ns_opt", pybert.t_ns[:len(p_tot)])
                pybert.plotdata.set_data("curs_amp", [0, curs_amp])
                curs_time = pybert.t_ns[curs_ix]
                pybert.plotdata.set_data("curs_ix", [curs_time, curs_time])
                fom_max = fom
                time.sleep(0.001)
            trials_run += 1
            if not trials_run % 100:
                pybert.status = f"Optimizing EQ...({100 * trials_run // n_trials}%)"
                time.sleep(0.001)
                if pybert.opt_thread.stopped():
                    pybert.status = "Optimization aborted by user."
                    raise RuntimeError("Optimization aborted by user.")

    for k, dfe_weight in enumerate(dfe_weights_best):  # pylint: disable=possibly-used-before-assignment
        dfe_taps[k].value = dfe_weight

    return (tx_weights_best, peak_mag_best, fom_max, True)  # pylint: disable=possibly-used-before-assignment
