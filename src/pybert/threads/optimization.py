"""
PyBERT Linear Equalization Optimizer

Original author: David Banas <capn.freako@gmail.com>

Original date: June 21, 2017

Copyright (c) 2017 David Banas; all rights reserved World wide.

TX, RX or co optimization are run in a separate thread to keep the gui responsive.
"""

import time

import optuna
from numpy import arange, array, convolve, delete, insert, ones, pi, prod, where, zeros
from numpy.fft import irfft, rfft
from scipy.interpolate import interp1d

from pychopmarg.optimize import mmse
from pychopmarg.noise import NoiseCalc

from pybert.common import Rvec
from pybert.models.tx_tap import TxTapTuner
from pybert.threads.stoppable import StoppableThread
from pybert.utility import make_ctle, calc_resps, add_ffe_dfe, get_dfe_weights, resize_zero_pad, safe_log10
from pybert.utility.ibisami import run_ami_model

optuna.logging.set_verbosity(optuna.logging.WARNING)  # PyBERT drives its own status/log reporting.

gDebugOptimize = False


def _suggest(trial: optuna.Trial, prefix: str, tuner) -> float:
    "Ask `trial` to suggest a value for `tuner`, dispatching on its declared type."
    name = f"{prefix}_{tuner.name}"
    if getattr(tuner, "is_int", False):
        step = max(1, round(tuner.step)) if tuner.step else 1
        return float(trial.suggest_int(name, round(tuner.min_val), round(tuner.max_val), step=step))
    return trial.suggest_float(name, tuner.min_val, tuner.max_val, step=tuner.step or None)


# pylint: disable=no-member
class OptThread(StoppableThread):
    "Used to run EQ optimization in its own thread, to preserve GUI responsiveness."

    def run(self):
        "Run the equalization optimization thread."

        pybert = self.pybert

        pybert.status = "Optimizing EQ..."
        time.sleep(0.001)

        try:
            tx_weights, rx_peaking, rx_weights, fom, valid, tx_ami_vals, rx_ami_vals = coopt(pybert)
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
        # `coopt()` already committed the winning AMI parameter values into
        # `pybert._tx_cfg`/`pybert._rx_cfg`; just refresh the tuner tables' displayed values.
        for k, val in enumerate(tx_ami_vals):
            pybert.tx_ami_tap_tuners[k].value = val
        for k, val in enumerate(rx_ami_vals):
            pybert.rx_ami_tap_tuners[k].value = val
        pybert.status = f"Finished. (SNR: {20 * safe_log10(fom):5.1f} dB)"


def mk_tap_weight_combs(tap_tuners: list[TxTapTuner]) -> list[Rvec]:
    """
    Make all tap weight combinations possible from a list of tap tuners.

    Args:
        tap_tuners: List of tap tuner control objects.

    Return:
        List of all possible tap weight combinations.

    Raises:
        ValueError: If empty list given as input.
        ValueError: If total number of combinations is too large.
    """

    if not tap_tuners:
        raise ValueError("Input list may not be empty!")

    # Check total number of combinations.
    n_combs = prod([int((tuner.max_val - tuner.min_val) / tuner.step + 1)
                    if tuner.enabled else 1
                    for tuner in tap_tuners])
    if n_combs > 1_000_000:
        raise ValueError(
            f"Total number of combinations ({int(n_combs // 1e6)} M) is too great!")

    # Prime recursive helper, then trim priming from results.
    rslts = _mk_tap_weight_combs([zeros(len(tap_tuners))], list(enumerate(tap_tuners)))
    rslts = list(filter(lambda xs: any(xs != 0.0), rslts))
    return rslts


def _mk_tap_weight_combs(weightss: list[Rvec], enumerated_tuners: list[tuple[int, TxTapTuner]]) -> list[Rvec]:
    """
    Recursive helper function.

    Args:
        weightss: The current list of tap weight combinations. (Supports recursion.)
        enumerated_tuners: List of pairs, each containing

            - the index of this tap in the list, and
            - this tap tuner.

    Return:
        List of all possible tap weight combinations.
    """

    # Check for end of recursion.
    if not enumerated_tuners:
        return weightss

    # Perform normal (i.e. - recursive) calculation.
    head, *tail = enumerated_tuners  # Pythonic list expansion to: first element, all the rest
    n, tuner = head
    if not tuner.enabled:
        return _mk_tap_weight_combs(weightss, tail)  # Skip this tap if its associated tuner is disabled.
    # Expand the current list by replicating each existing item several times,
    # according to a sweep of the current tap being considered.
    weight_vals = arange(tuner.min_val, tuner.max_val + tuner.step, tuner.step)
    new_weightss = []
    for weights in weightss:
        for val in weight_vals:
            weights[n] = val
            new_weightss.append(weights.copy())
    return _mk_tap_weight_combs(new_weightss, tail)


def coopt(pybert) -> tuple[list[float], float, list[float], float, bool, Rvec, Rvec]:  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """
    Co-optimize the Tx/Rx linear equalization, assuming ideal bounded DFE.

    When Tx and/or Rx is IBIS-AMI, candidate parameter values come from a
    TPE-guided (Optuna) search, instead of an exhaustive grid: each trial
    means a real `AMI_Init()` DLL call, orders of magnitude slower than the
    pure-native path's `numpy.convolve()`-based evaluation, so a Bayesian
    search that spends its budget where it's informative wins over a grid
    that wastes most of it on uninformative points (and has no way to know,
    e.g., that a parameter gated "Off" by a sibling mode-selector is a no-op
    to sweep). The pure-native path (both sides native) is untouched: it's
    already fast and exact (closed-form MMSE, or a small exhaustive grid).

    Args:
        pybert(PyBERT): The PyBERT instance on which to perform co-optimization.

    Returns:
        A tuple containing

            - the optimum native Tx FFE tap weights (empty, if Tx is IBIS-AMI),
            - the optimum native Rx CTLE peaking (0, if Rx is IBIS-AMI),
            - the optimum native Rx FFE tap weights (empty, if Rx is IBIS-AMI),
            - the figure of merit for the returned settings,
            - the status of the optimization attempt (`True` = success),
            - the optimum Tx IBIS-AMI parameter values (empty, if Tx is native), and
            - the optimum Rx IBIS-AMI parameter values (empty, if Rx is native).

        When Tx and/or Rx is IBIS-AMI, the winning parameter values have
        already been committed to `pybert._tx_cfg`/`pybert._rx_cfg` (the
        live objects `my_run_simulation()` reads), so no separate "Use EQ"
        step is required for them.

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
    num_levels = pybert.mod_type_ + 2
    tx_use_ami = pybert.tx_use_ami
    rx_use_ami = pybert.rx_use_ami

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
    ts = t[1]
    nspui = pybert.nspui
    f = pybert.f
    _, p_chnl, _ = calc_resps(t, h_chnl, ui, f)
    pybert.plotdata.set_data("p_chnl", p_chnl)

    # Calculate `f_t` and interpolated channel frequency response.
    # (Needed below, for MMSE scoring.)
    dt = t[1] - t[0]            # `t` assumed uniformly sampled throughout.
    fN = 0.5 / dt               # Nyquist frequency
    f0 = 100e6                  # fundamental frequency
    f_t = arange(0, fN + f0 / 2, f0)  # "+ f0 / 2", to ensure `fN` gets included.
    _t = array([n * dt for n in range((len(f_t) - 1) * 2)])
    krnl = interp1d(f, pybert.chnl_H, bounds_error=False, fill_value=0)
    chnl_H = krnl(f_t)

    # An AMI model's own Init()-returned impulse response slots directly into the same
    # convolution cascade used for native EQ (both are just LTI impulse responses), so a
    # Tx/Rx AMI candidate is evaluated by really invoking `AMI_Init()`, in place of the
    # closed-form tap/CTLE math. See this feature's design doc for the full rationale.
    # Stand-in "no channel" for Tx-AMI candidates: a unit impulse on the same sample grid as
    # `h_chnl` (PyAMI's generic response post-processing needs at least `nspui` samples to work).
    identity_h = zeros(len(h_chnl))
    identity_h[0] = 1.0

    tx_curs_pos = max(0, -tx_taps[0].pos)  # list position at which to insert the native Tx cursor tap
    effective_use_mmse = pybert.use_mmse and not rx_use_ami
    n_rx_weights = len(rx_taps)
    rx_weightss = None  # Only used by the native, non-MMSE Rx-FFE sub-solve, below.
    if not rx_use_ami and not effective_use_mmse:
        try:
            rx_weightss = mk_tap_weight_combs(rx_taps)
            if not rx_weightss:  # Trap the "null FFE" case.
                rx_weightss = [array([0.0] * rx_n_pre + [1.0] + [0.0] * (rx_n_taps - rx_n_pre - 1))]
        except ValueError as err:
            raise RuntimeError(
                "\n".join([
                    f"{err}",
                    "Sorry, that's more Rx FFE tap weight combinations than I can handle.",
                    "I had to abort the EQ optimization in your stead.",
                ])) from err
    dfe_weights = zeros(len(dfe_taps))  # Used by the native, non-MMSE Rx-FFE sub-solve, below.

    def score_candidate(p_ctle_out: Rvec, ctle_H, h_tx: Rvec):
        """
        Given one Tx/Rx-CTLE candidate's impulse responses, find the best achievable
        Rx FFE/DFE design for it: an exact closed-form MMSE solve (native Rx, `use_mmse`),
        a small exhaustive grid (native Rx, non-MMSE) -- both cheap, no extra AMI calls --
        or, when Rx is AMI (which implements its own EQ internally), nothing to design at
        all: just the cursor/ISI figure of merit straight off the cascade's pulse response.

        Returns:
            A tuple of (figure of merit, Rx FFE tap weights, DFE tap weights, total pulse response).
        """
        p_tx = convolve(p_ctle_out, h_tx)
        p_tx = resize_zero_pad(p_tx, len(_t))
        if effective_use_mmse:
            curs_ix = where(p_tx == max(p_tx))[0][0]
            tx_H = rfft(resize_zero_pad(h_tx, len(_t)))
            noise_calc = NoiseCalc(
                num_levels, ui, curs_ix, _t, p_tx, [], f_t,
                tx_H, chnl_H, ones(len(f_t)), ctle_H,
                0.0, 0.5, 25, 0.0, 0.0
            )
            mmse_rslts = mmse(
                noise_calc, rx_n_taps, rx_n_pre, n_dfe_taps, pybert.rlm, pybert.mod_type_ + 2,
                array(list(map(lambda t: t.min_val, dfe_taps[:n_dfe_taps]))), array(list(map(lambda t: t.max_val, dfe_taps[:n_dfe_taps]))),
                array(list(map(lambda t: t.min_val, rx_taps[:rx_n_taps]))), array(list(map(lambda t: t.max_val, rx_taps[:rx_n_taps]))))
            rx_weights_better = mmse_rslts["rx_taps"]
            dfe_weights_better = mmse_rslts["dfe_tap_weights"]
            fom_better = mmse_rslts["fom"]
            try:
                p_tot = resize_zero_pad(add_ffe_dfe(rx_weights_better, dfe_weights_better, nspui, p_tx),
                                         nspui * (n_rx_weights + 5))
            except ValueError:  # Flags obviously non-optimum case.
                return -1000., zeros(n_rx_weights), zeros(len(dfe_taps)), p_tx
            return fom_better, rx_weights_better, dfe_weights_better, p_tot
        if rx_use_ami:  # Rx implements its own EQ internally; nothing left to design.
            curs_ix = where(p_tx == max(p_tx))[0][0]
            curs_amp = p_tx[curs_ix]
            n_pre_isi = curs_ix // nspui
            isi_sum = sum(abs(p_tx[curs_ix - n_pre_isi * nspui::nspui])) - abs(curs_amp)
            return curs_amp / isi_sum, zeros(rx_n_taps), zeros(len(dfe_taps)), p_tx
        # exhaustive sweep of Rx FFE tap weight combinations
        assert rx_weightss is not None  # Guaranteed by the `not rx_use_ami and not effective_use_mmse` setup, above.
        fom_better = -1000.
        rx_weights_better = zeros(n_rx_weights)
        dfe_weights_better = dfe_weights
        p_tot = p_tx
        for rx_weights in rx_weightss:
            try:
                p_tot_cand = add_ffe_dfe(rx_weights, array(get_dfe_weights(dfe_taps, p_tx, nspui)), nspui, p_tx)
            except ValueError:  # Flags obviously non-optimum case.
                continue
            curs_ix = where(p_tot_cand == max(p_tot_cand))[0][0]
            curs_amp = p_tot_cand[curs_ix]
            n_pre_isi = curs_ix // nspui
            isi_sum = sum(abs(p_tot_cand[curs_ix - n_pre_isi * nspui::nspui])) - abs(curs_amp)
            fom = curs_amp / isi_sum
            if fom > fom_better:
                rx_weights_better = rx_weights
                dfe_weights_better = dfe_weights
                fom_better = fom
                p_tot = p_tot_cand
        return fom_better, rx_weights_better, dfe_weights_better, p_tot

    def native_ctle_candidate(peak_mag: float):
        "Build (p_ctle_out, ctle_H) for one native CTLE peaking-magnitude value."
        _, H_ctle = make_ctle(rx_bw, peak_freq, peak_mag, w_ctle)
        _h_ctle = irfft(H_ctle)
        krnl_ctle = interp1d(t_ctle, _h_ctle, bounds_error=False, fill_value=0)
        h_ctle = krnl_ctle(t[:max_len])
        h_ctle *= sum(_h_ctle) / sum(h_ctle)  # type: ignore
        p_ctle_out = convolve(p_chnl, h_ctle)[:len(p_chnl)]
        ctle_H = rfft(resize_zero_pad(h_ctle, len(_t)))
        return p_ctle_out, ctle_H

    def coopt_native():
        "Pure-native (both Tx and Rx native) co-optimization: exhaustive grid, exact sub-solves."
        if pybert.ctle_enable_tune:
            peak_mags = arange(min_mag, max_mag + step_mag, step_mag)
        else:
            peak_mags = array([0])
        rx_candidates = [(peak_mag, *native_ctle_candidate(peak_mag)) for peak_mag in peak_mags]

        try:
            tx_weightss = mk_tap_weight_combs(pybert.tx_tap_tuners)
        except ValueError as err:
            raise RuntimeError(
                "\n".join([
                    f"{err}",
                    "Sorry, that's more Tx tap weight combinations than I can handle.",
                    "I had to abort the EQ optimization in your stead.",
                ])) from err
        tx_weightss = list(map(lambda ws: insert(ws, tx_curs_pos, 1 - sum(abs(ws))), tx_weightss))
        tx_candidates = []
        for tx_weights in tx_weightss:
            # sum = concatenate
            h_tx = array(sum([[tx_weight] + [0] * (nspui - 1) for tx_weight in tx_weights], []))
            tx_candidates.append((tx_weights, h_tx))

        n_enabled_tx = len(list(filter(lambda t: t.enabled, tx_taps)))
        n_enabled_rx = len(list(filter(lambda t: t.enabled, rx_taps)))
        n_trials = len(rx_candidates) * len(tx_candidates)
        trials_run_inc = n_trials // 100 or 1
        pybert.log("\n".join([
            "Optimizing linear EQ...",
            f"\tOversampling factor: {nspui}",
            f"\tTx equalization: native ({n_enabled_tx} enabled tap(s), cursor at {tx_curs_pos})",
            f"\tRx equalization: native ({n_enabled_rx} enabled FFE tap(s))",
            f"\tRunning {n_trials} trials.",
            ""]))

        fom_max = -1000.
        peak_mag_best = 0.
        trials_run = 0
        rx_weights_best = zeros(n_rx_weights)
        dfe_weights_best = zeros(len(dfe_taps))
        tx_weights_best = [0.0] * len(tx_taps)
        del tx_weights_best[tx_curs_pos]
        for peak_mag, p_ctle_out, ctle_H in rx_candidates:
            for tx_weights, h_tx in tx_candidates:
                fom, rx_weights, dfe_weights_c, p_tot = score_candidate(p_ctle_out, ctle_H, h_tx)
                trials_run += 1
                _report_progress(pybert, trials_run, n_trials, trials_run_inc)
                if fom > fom_max:
                    rx_weights_best = rx_weights.copy()
                    dfe_weights_best = dfe_weights_c.copy()
                    tx_weights_best = list(delete(tx_weights, tx_curs_pos))
                    peak_mag_best = peak_mag
                    _update_plotdata(pybert, p_tot, nspui)
                    fom_max = fom
                    time.sleep(0.001)

        return tx_weights_best, peak_mag_best, rx_weights_best, fom_max, dfe_weights_best, array([]), array([])

    def coopt_ami():
        """
        AMI-active (Tx and/or Rx IBIS-AMI) co-optimization: a TPE (Optuna) search over
        every *enabled* parameter, on whichever side(s) are active -- Tx AMI parameters,
        Rx AMI parameters, and/or (if the other side is native) native Tx tap weights /
        Rx CTLE peaking, all suggested jointly, per trial, from one unified search space.
        Each trial invokes the real AMI model(s) via `run_ami_model()`, so the search is
        driven by Optuna's TPE sampler instead of an exhaustive grid.
        """
        tx_ami_tuners = pybert.tx_ami_tap_tuners if tx_use_ami else []
        rx_ami_tuners = pybert.rx_ami_tap_tuners if rx_use_ami else []
        tx_returns_impulse = True
        rx_returns_impulse = True
        tx_fixed_h = None
        rx_fixed_p_ctle_out = None
        if tx_use_ami:
            tx_returns_impulse = bool(pybert._tx_cfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"]))
            if not tx_returns_impulse:
                if any(tuner.enabled for tuner in tx_ami_tuners):
                    pybert.log(
                        "This AMI model only supports GetWave(); Tx parameter optimization isn't supported yet"
                        " -- configure it manually via the 'Configure' button.")
                tx_fixed_h = array([1.0])  # GetWave-only: can't sweep it; pass the signal through unmodified.
        if rx_use_ami:
            rx_returns_impulse = bool(pybert._rx_cfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"]))
            if not rx_returns_impulse:
                if any(tuner.enabled for tuner in rx_ami_tuners):
                    pybert.log(
                        "This AMI model only supports GetWave(); Rx parameter optimization isn't supported yet"
                        " -- configure it manually via the 'Configure' button.")
                rx_fixed_p_ctle_out = p_chnl  # GetWave-only: can't sweep it; fall back to the unequalized channel.

        # Total search-space dimensionality -- if zero (nothing enabled/sweepable anywhere),
        # there's nothing for a sampler to search: skip Optuna and run one fixed evaluation.
        n_dims = 0
        if tx_use_ami:
            if tx_returns_impulse:
                n_dims += sum(1 for tuner in tx_ami_tuners if tuner.enabled)
        else:
            n_dims += sum(1 for tuner in tx_taps if tuner.enabled)
        if rx_use_ami:
            if rx_returns_impulse:
                n_dims += sum(1 for tuner in rx_ami_tuners if tuner.enabled)
        elif pybert.ctle_enable_tune:
            n_dims += 1

        if n_dims == 0:
            n_trials = 1
            study = None
        else:
            n_trials = pybert.ami_opt_trials
            sampler = optuna.samplers.TPESampler(seed=(pybert.seed or None))
            study = optuna.create_study(direction="maximize", sampler=sampler)

        n_enabled_tx = len(list(filter(lambda t: t.enabled, (tx_ami_tuners if tx_use_ami else tx_taps))))
        n_enabled_rx = len(list(filter(lambda t: t.enabled, (rx_ami_tuners if rx_use_ami else rx_taps))))
        tx_desc = (f"IBIS-AMI ({n_enabled_tx} enabled param(s))" if tx_use_ami
                   else f"native ({n_enabled_tx} enabled tap(s), cursor at {tx_curs_pos})")
        rx_desc = (f"IBIS-AMI ({n_enabled_rx} enabled param(s))" if rx_use_ami
                   else f"native ({n_enabled_rx} enabled FFE tap(s))")
        pybert.log("\n".join([
            "Optimizing linear EQ...",
            f"\tOversampling factor: {nspui}",
            f"\tTx equalization: {tx_desc}",
            f"\tRx equalization: {rx_desc}",
            "\tSearch: " + ("TPE (Optuna)" if study is not None else "single fixed evaluation (nothing enabled)"),
            f"\tRunning {n_trials} trial(s).",
            ""]))

        fom_max = -1000.
        peak_mag_best = 0.
        trials_run = 0
        rx_weights_best = zeros(n_rx_weights)
        dfe_weights_best = zeros(len(dfe_taps))
        tx_weights_best: list = []
        tx_ami_best: Rvec = array([])
        rx_ami_best: Rvec = array([])
        trials_run_inc = n_trials // 100 or 1
        for _ in range(n_trials):
            trial = study.ask() if study is not None else None

            # --- Tx side: real AMI_Init(), or native FFE tap synthesis. ---
            if tx_use_ami:
                if tx_returns_impulse:
                    tx_vals = []
                    for tuner in tx_ami_tuners:
                        val = _suggest(trial, "tx", tuner) if tuner.enabled else tuner.value
                        if tuner.enabled:
                            pybert._tx_cfg.set_param_val(list(tuner.branch_names), val)
                        tx_vals.append(val)
                    _, _, h_tx, _, _, _ = run_ami_model(
                        pybert.tx_dll_file, pybert._tx_cfg, False, ui, ts, identity_h, zeros(1))
                    tx_ami_vals = array(tx_vals)
                else:
                    h_tx = tx_fixed_h
                    tx_ami_vals = array([tuner.value for tuner in tx_ami_tuners])
                tx_weights = None
            else:
                ws = array([_suggest(trial, "tx", tuner) if tuner.enabled else 0.0 for tuner in tx_taps])
                ws_full = insert(ws, tx_curs_pos, 1 - sum(abs(ws)))
                h_tx = array(sum([[tw] + [0] * (nspui - 1) for tw in ws_full], []))
                tx_weights = ws
                tx_ami_vals = array([])

            # --- Rx side: real AMI_Init(), or native CTLE. ---
            if rx_use_ami:
                if rx_returns_impulse:
                    rx_vals = []
                    for tuner in rx_ami_tuners:
                        val = _suggest(trial, "rx", tuner) if tuner.enabled else tuner.value
                        if tuner.enabled:
                            pybert._rx_cfg.set_param_val(list(tuner.branch_names), val)
                        rx_vals.append(val)
                    _, _, _, out_h, _, _ = run_ami_model(
                        pybert.rx_dll_file, pybert._rx_cfg, False, ui, ts, h_chnl, zeros(1))
                    _, p_ctle_out, _ = calc_resps(t, out_h, ui, f)
                    rx_ami_vals = array(rx_vals)
                else:
                    p_ctle_out = rx_fixed_p_ctle_out
                    rx_ami_vals = array([tuner.value for tuner in rx_ami_tuners])
                ctle_H = None
                peak_mag = 0.0
            else:
                peak_mag = trial.suggest_float("rx_peak_mag", min_mag, max_mag, step=step_mag) \
                    if pybert.ctle_enable_tune else 0.0
                p_ctle_out, ctle_H = native_ctle_candidate(peak_mag)
                rx_ami_vals = array([])

            fom, rx_weights, dfe_weights_c, p_tot = score_candidate(p_ctle_out, ctle_H, h_tx)
            if study is not None:
                study.tell(trial, fom)
            trials_run += 1
            _report_progress(pybert, trials_run, n_trials, trials_run_inc)
            if fom > fom_max:
                rx_weights_best = rx_weights.copy()
                dfe_weights_best = dfe_weights_c.copy()
                tx_weights_best = list(tx_weights) if tx_weights is not None else []
                tx_ami_best = tx_ami_vals.copy()
                peak_mag_best = peak_mag
                rx_ami_best = rx_ami_vals.copy()
                _update_plotdata(pybert, p_tot, nspui)
                fom_max = fom
                time.sleep(0.001)

        return tx_weights_best, peak_mag_best, rx_weights_best, fom_max, dfe_weights_best, tx_ami_best, rx_ami_best

    if tx_use_ami or rx_use_ami:
        rslts = coopt_ami()
    else:
        rslts = coopt_native()
    tx_weights_best, peak_mag_best, rx_weights_best, fom_max, dfe_weights_best, tx_ami_best, rx_ami_best = rslts

    for k, dfe_weight in enumerate(dfe_weights_best):
        dfe_taps[k].value = dfe_weight

    # Commit the winning AMI parameter values. `_tx_cfg`/`_rx_cfg` are exactly the live
    # objects `my_run_simulation()` reads, so this step *is* the "Use EQ" commit, for AMI --
    # unlike native EQ, no separate copy-into-live-traits step is needed.
    if tx_use_ami and len(tx_ami_best):
        for tuner, val in zip(pybert.tx_ami_tap_tuners, tx_ami_best):
            if tuner.enabled:
                pybert._tx_cfg.set_param_val(list(tuner.branch_names), float(val))
    if rx_use_ami and len(rx_ami_best):
        for tuner, val in zip(pybert.rx_ami_tap_tuners, rx_ami_best):
            if tuner.enabled:
                pybert._rx_cfg.set_param_val(list(tuner.branch_names), float(val))

    return tx_weights_best, peak_mag_best, list(rx_weights_best), fom_max, True, tx_ami_best, rx_ami_best


def _report_progress(pybert, trials_run: int, n_trials: int, trials_run_inc: int) -> None:
    "Shared progress/abort-check, called once per evaluated candidate."
    if not trials_run % trials_run_inc:
        pybert.status = f"Optimizing EQ...({100 * trials_run // n_trials}%)"
        time.sleep(0.001)
        if pybert.opt_thread and pybert.opt_thread.stopped():
            pybert.status = "Optimization aborted by user."
            raise RuntimeError("Optimization aborted by user.")


def _update_plotdata(pybert, p_tot: Rvec, nspui: int) -> None:
    "Refresh the Optimizer tab's live plot with the current best candidate's pulse response."
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

