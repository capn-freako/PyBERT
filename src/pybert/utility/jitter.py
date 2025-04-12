"""
Jitter utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

from typing import Optional

import numpy as np
from numpy import (  # type: ignore
    argmax, array, concatenate, diag, diff, flip,
    histogram, mean, ones, real, reshape, resize, sign,
    sort, sqrt, where, zeros
)
from numpy.fft import fft, ifft  # type: ignore
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from ..common import Rvec

from .math import gaus_pdf
from .sigproc import moving_average

debug          = False


def find_crossing_times(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    t: Rvec,
    x: Rvec,
    min_delay: float = 0.0,
    rising_first: bool = True,
    min_init_dev: float = 0.1,
    thresh: float = 0.0,
) -> Rvec:
    """
    Finds the threshold crossing times of the input signal.

    Args:
        t: Vector of sample times. Intervals do NOT need to be uniform.
        x: Sampled input vector.

    Keyword Args:
        min_delay: Minimum delay required, before allowing crossings.
            (Helps avoid false crossings at beginning of signal.)
            Default: 0
        rising_first: When True, start with the first rising edge found.
            When this option is True, the first rising edge crossing is the first crossing returned.
            This is the desired behavior for PyBERT, because we always
            initialize the bit stream with [0, 0, 1, 1], in order to
            provide a known synchronization point for jitter analysis.
            Default: True
        min_init_dev: The minimum initial deviation from zero,
            which must be detected before searching for crossings.
            Normalized to maximum input signal magnitude.
            Default: 0.1
        thresh: Vertical crossing threshold.

    Returns:
        xing_times: Array of signal threshold crossing times.
    """

    if len(t) != len(x):
        raise ValueError(f"len(t) ({len(t)}) and len(x) ({len(x)}) need to be the same.")

    t = array(t)
    x = array(x)

    try:
        max_mag_x = np.max(abs(x))
    except Exception:  # pylint: disable=broad-exception-caught
        print("len(x):", len(x))
        raise
    min_mag_x = min_init_dev * max_mag_x
    i = 0
    while abs(x[i]) < min_mag_x:
        i += 1
        if i >= len(x):
            raise RuntimeError("Input signal minimum deviation not detected!")
    x = x[i:] - thresh
    t = t[i:]

    sign_x = sign(x)
    sign_x = where(sign_x, sign_x, ones(len(sign_x)))  # "0"s can produce duplicate xings.
    diff_sign_x = diff(sign_x)
    xing_ix = where(diff_sign_x)[0]
    xings = [t[i] + (t[i + 1] - t[i]) * x[i] / (x[i] - x[i + 1]) for i in xing_ix]

    if not xings:
        return array([])

    i = 0
    if min_delay:
        if min_delay >= xings[-1]:
            raise RuntimeError(f"min_delay ({min_delay}) must be less than last crossing time ({xings[-1]}).")
        while xings[i] < min_delay:
            i += 1

    if debug:
        print(f"min_delay: {min_delay}")
        print(f"rising_first: {rising_first}")
        print(f"i: {i}")
        print(f"max_mag_x: {max_mag_x}")
        print(f"min_mag_x: {min_mag_x}")
        print(f"xings[0]: {xings[0]}")
        print(f"xings[i]: {xings[i]}")

    try:
        if rising_first and diff_sign_x[xing_ix[i]] < 0.0:
            i += 1
    except Exception:  # pylint: disable=broad-exception-caught
        print("len(diff_sign_x):", len(diff_sign_x))
        print("len(xing_ix):", len(xing_ix))
        print("i:", i)
        raise

    return array(xings[i:])


def find_crossings(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    t: Rvec,
    x: Rvec,
    amplitude: float = 1.0,
    min_delay: float = 0.0,
    rising_first: bool = True,
    min_init_dev: float = 0.1,
    mod_type: int = 0
) -> Rvec:
    """
    Find the crossing times in a signal, according to the modulation type.

    Args:
        t: The times associated with each signal sample.
        x: The signal samples.

    Keyword Args:
        amplitude: The nominal signal amplitude.
            (Used for determining thresholds, in the case of some modulation types.)
            Default: 1.0
        min_delay(float): The earliest possible sample time we want returned.
            Default: 0
        rising_first: When True, start with the first rising edgefound.
            When this option is True, the first rising edge
            crossing is the first crossing returned. This is the desired
            behavior for PyBERT, because we always initialize the bit
            stream with [0, 0, 1, 1], in order to provide a known
            synchronization point for jitter analysis.
            Default: True
        min_init_dev: The minimum initial deviation from zero,
            which must be detected before searching for crossings.
            Normalized to maximum input signal magnitude.
            Default: 0.1
        mod_type: The modulation type. Allowed values are:
            {0: NRZ, 1: Duo-binary, 2: PAM-4}
            Default: 0

    Returns:
        xing_times: The signal threshold crossing times.
    """

    if mod_type not in [0, 1, 2]:
        raise ValueError(f"ERROR: pybert_util.find_crossings(): Unknown modulation type: {mod_type}")

    xings = []
    if mod_type == 0:  # NRZ
        xings.append(
            find_crossing_times(t, x, min_delay=min_delay, rising_first=rising_first, min_init_dev=min_init_dev)
        )
    elif mod_type == 1:  # Duo-binary
        xings.append(
            find_crossing_times(
                t,
                x,
                min_delay=min_delay,
                rising_first=rising_first,
                min_init_dev=min_init_dev,
                thresh=(-0.5 * amplitude),
            )
        )
        xings.append(
            find_crossing_times(
                t,
                x,
                min_delay=min_delay,
                rising_first=rising_first,
                min_init_dev=min_init_dev,
                thresh=(0.5 * amplitude),
            )
        )
    elif mod_type == 2:  # PAM-4 (Enabling the +/-0.67 cases yields multiple ideal crossings at the same edge.)
        xings.append(
            find_crossing_times(
                t,
                x,
                min_delay=min_delay,
                rising_first=rising_first,
                min_init_dev=min_init_dev,
                thresh=(0.0 * amplitude),
            )
        )
    else:
        raise ValueError(f"Unknown modulation type: {mod_type}")

    return sort(concatenate(xings))


def calc_jitter(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements,too-many-positional-arguments
    ui: float, nui: int, pattern_len: int, ideal_xings: Rvec, actual_xings: Rvec,
    rel_thresh: float = 3.0, num_bins: int = 101,
    zero_mean: bool = True, dbg_obj: Optional[object] = None, smooth_width: int = 5
) -> tuple[Rvec, Rvec, float, float, float, float, float, float,
           Rvec, Rvec, Rvec, Rvec, Rvec, Rvec, Rvec, Rvec, float, float]:
    """
    Calculate the jitter in a set of actual zero crossings,
    given the ideal crossings and unit interval.

    Args:
        ui: The nominal unit interval (s).
        nui: The number of unit intervals spanned by the input signal.
        pattern_len: The number of unit intervals, before input symbol stream repeats.
        ideal_xings: The ideal zero crossing locations of the edges (s).
        actual_xings: The actual zero crossing locations of the edges (s).

    Keyword Args:
        rel_thresh: The threshold for determining periodic jitter spectral components (sigma).
            Default: 3.0
        num_bins: The number of bins to use, when forming histograms.
            Default: 101
        zero_mean: Force the mean jitter to zero, when True.
            Default: True
        dbg_obj: Object for stashing debugging info.
            Default: None
        smooth_width: Width of smoothing window to use when calculating moving averages.
            Default: 5

    Returns:
        ( Jtot: The total jitter.
        , times: The times (taken from 'ideal_xings') corresponding to the returned jitter values.
        , jISI: The peak to peak jitter due to intersymbol interference (ISI).
        , jDCD: The peak to peak jitter due to duty cycle distortion (DCD).
        , jPj: The peak to peak jitter due to uncorrelated periodic sources (Pj).
        , jRj: The standard deviation of the jitter due to uncorrelated unbounded random sources (Rj).
        , jPjDD: Dual-Dirac peak to peak jitter.
        , jRjDD: Dual-Dirac random jitter.
        , Jind: The data independent jitter.
        , thresh: Threshold for determining periodic components.
        , Stot: The spectral magnitude of the total jitter.
        , Sind: The spectral magnitude of the data independent jitter.
        , freqs: The frequencies corresponding to the spectrum components.
        , histTOT: The smoothed histogram of the total jitter.
        , histIND: The smoothed histogram of the data-independent jitter.
        , centers: The bin center values for both histograms.
        , mu_pos: The mean of the Gaussian distribution best fitted to the right tail.
        , mu_neg: The mean of the Gaussian distribution best fitted to the left tail.
        )

    Raises:
        ValueError: If input checking fails, or curve fitting goes awry.

    Notes:
        1. The actual crossings should arrive pre-aligned to the ideal crossings.
        And both should start near zero time.
    """
    # Check inputs.
    if not ideal_xings.all():
        raise ValueError("calc_jitter(): zero length ideal crossings vector received!")
    if not actual_xings.all():
        raise ValueError("calc_jitter(): zero length actual crossings vector received!")

    num_patterns = nui // pattern_len
    if num_patterns == 0:
        raise ValueError("\n".join([
            "Need at least one full pattern repetition!",
            f"(pattern_len: {pattern_len}; nui: {nui})",]))
    xings_per_pattern = where(ideal_xings > (pattern_len * ui))[0][0]
    if xings_per_pattern % 2 or not xings_per_pattern:
        raise ValueError("\n".join([
            "pybert.utility.calc_jitter(): Odd number of (or, no) crossings per pattern detected!",
            f"xings_per_pattern: {xings_per_pattern}",
            f"min(ideal_xings): {min(ideal_xings)}",]))

    # Assemble the TIE track.
    i = 0
    jitterL = []
    t_jitter = []
    skip_next_ideal_xing = False
    for ideal_xing in ideal_xings:
        if skip_next_ideal_xing:
            t_jitter.append(ideal_xing)
            skip_next_ideal_xing = False
            continue
        # Confine our attention to those actual crossings occuring
        # within the interval [-UI/2, +UI/2] centered around the
        # ideal crossing.
        min_t = ideal_xing - ui / 2.0
        max_t = ideal_xing + ui / 2.0
        while i < len(actual_xings) and actual_xings[i] < min_t:
            i += 1
        if i == len(actual_xings):  # We've exhausted the list of actual crossings; we're done.
            break
        if actual_xings[i] > max_t:  # Means the xing we're looking for didn't occur, in the actual signal.
            jitterL.append( 3.0 * ui / 4.0)  # Pad the jitter w/ alternating +/- 3UI/4.  # noqa: E201
            jitterL.append(-3.0 * ui / 4.0)  # (Will get pulled into [-UI/2, UI/2], later.
            skip_next_ideal_xing = True  # If we missed one, we missed two.
        else:  # Noise may produce several crossings.
            xings = []  # We find all those within the interval [-UI/2, +UI/2]
            j = i  # centered around the ideal crossing, and take the average.
            while j < len(actual_xings) and actual_xings[j] <= max_t:
                xings.append(actual_xings[j])
                j += 1
            tie = mean(xings) - ideal_xing
            jitterL.append(tie)
        t_jitter.append(ideal_xing)
    jitter = array(jitterL)

    # ToDo: Report this in the status bar.
    if len(jitter) == 0 or len(t_jitter) == 0:
        print("No crossings found!", flush=True)

    if debug:
        print("mean(jitter):", mean(jitter))
        print("len(jitter):", len(jitter))

    if zero_mean:
        jitter -= mean(jitter)

    # Do the jitter decomposition.
    # - Separate the rising and falling edges, shaped appropriately for averaging over the pattern period.
    tie_risings  = jitter.take(list(range(0, len(jitter), 2)))  # type: ignore
    tie_fallings = jitter.take(list(range(1, len(jitter), 2)))  # type: ignore
    tie_risings.resize(num_patterns * xings_per_pattern // 2, refcheck=False)  # type: ignore
    tie_fallings.resize(num_patterns * xings_per_pattern // 2, refcheck=False)  # type: ignore
    tie_risings  = reshape(tie_risings, (num_patterns, xings_per_pattern // 2))
    tie_fallings = reshape(tie_fallings, (num_patterns, xings_per_pattern // 2))

    # - Use averaging to remove the uncorrelated components, before calculating data dependent components.
    tie_risings_ave  = tie_risings.mean(axis=0)
    tie_fallings_ave = tie_fallings.mean(axis=0)
    isi = max(tie_risings_ave.ptp(), tie_fallings_ave.ptp())
    isi = min(isi, ui)  # Cap the ISI at the unit interval.
    dcd = abs(mean(tie_risings_ave) - mean(tie_fallings_ave))

    # - Subtract the data dependent jitter from the original TIE track, in order to yield the data independent jitter.
    _jitter = jitter.copy()
    # "refcheck=False": accommodating Tox:
    _jitter.resize(num_patterns * xings_per_pattern, refcheck=False)  # type: ignore
    tie_ave = resize(reshape(_jitter, (num_patterns, xings_per_pattern)).mean(axis=0), len(jitter))
    tie_ind = jitter - tie_ave
    if zero_mean:
        tie_ind -= mean(tie_ind)

    # - Calculate the total and data-independent jitter spectrums, for display purposes only.
    # -- Calculate the relevant time/frequency vectors.
    osf = 1                                             # jitter oversampling factor
    t0  = ui / osf                                      # jitter sampling period
    t   = array([n * t0 for n in range(nui * osf)])  # jitter samples time vector
    f0  = 1.0 / (ui * nui)                              # jitter samples fundamental frequency
    _f   = [n * f0 for n in range(len(t) // 2)]          # [0:f0:fNyquist)
    f   = array(_f + [1 / (2 * t0)] + list(-1 * flip(array(_f[1:]))))  # [0:f0:fN) ++ [fN:-f0:0)
    half_len = len(f) // 2                              # for spectrum plotting convenience

    # -- Make TIE vector uniformly sampled in time, via interpolation, for use as input to `fft()`.
    # spl = UnivariateSpline(t_jitter, jitter)  # Way of the future, but does funny things. :(
    try:
        spl = interp1d(t_jitter, jitter, bounds_error=False, fill_value="extrapolate")
        tie_interp = spl(t)
        y = fft(tie_interp)
        jitter_spectrum = abs(y[:half_len])
    except Exception as err:  # pylint: disable=broad-exception-caught
        print(f"t_jitter: {t_jitter}")
        print(f"jitter: {jitter}")
        print(f"Error calculating data dependent TIE: {err}", flush=True)
        jitter_spectrum = zeros(half_len)
    jitter_freqs    = f[:half_len]

    # -- Repeat for data-independent jitter.
    try:
        spl = interp1d(t_jitter, tie_ind, bounds_error=False, fill_value="extrapolate")
        tie_ind_interp = spl(t)
        y = fft(tie_ind_interp)
        y_mag = abs(y)
        tie_ind_spectrum = y_mag[:half_len]
    except Exception as err:  # pylint: disable=broad-exception-caught
        print(f"t_jitter: {t_jitter}")
        print(f"tie_ind: {tie_ind}")
        print(f"Error calculating data independent TIE: {err}", flush=True)
        y = zeros(half_len)  # type: ignore
        y_mag = zeros(half_len)
        tie_ind_spectrum = zeros(half_len)

    # -- Perform spectral extraction of Pj from the data independent jitter,
    # -- using a threshold based on a moving average to identify the periodic components.
    win_width = 100
    y_mean    = moving_average(y_mag,                 n=win_width)
    y_var     = moving_average((y_mag - y_mean) ** 2, n=win_width)
    y_sigma   = sqrt(y_var)
    thresh    = y_mean + rel_thresh * y_sigma
    y_per     = where(y_mag > thresh, y, zeros(len(y)))  # Periodic components are those lying above the threshold.
    y_rnd     = where(y_mag > thresh, zeros(len(y)), y)  # Random components are those lying below.
    tie_per   = real(ifft(y_per))
    pj        = np.ptp(tie_per)
    tie_rnd   = real(ifft(y_rnd))
    rj        = sqrt(mean((tie_rnd - tie_rnd.mean())**2))

    # -- Do dual Dirac fitting of the data-independent jitter histogram, to determine Pj/Rj.
    # --- Generate a smoothed version of the TIE histogram, for better peak identification.
    # --- (Have to work in ps when curve fitting, or `curve_fit()` blows up.)
    use_my_hist = True  # False will yield misleading jitter distribution plots!

    def my_hist(x, density=False):
        """
        Calculates the probability *mass* function (PMF) of the input vector
        (or, the probability *density* function (PDF) if so directed),
        enforcing an output range of [-UI/2, +UI/2], sweeping everything in
        [-UI, -UI/2] into the first bin, and everything in [UI/2, UI]
        into the last bin.
        """
        bin_edges   = array([-ui] + [-ui / 2.0 + i * ui / (num_bins - 2) for i in range(num_bins - 1)] + [ui])
        bin_centers = [-ui / 2] + list((bin_edges[1:-2] + bin_edges[2:-1]) / 2) + [ui / 2]
        hist, _     = histogram(x, bin_edges)
        hist        = hist / hist.sum()  # PMF
        if density:
            hist /= diff(bin_edges)
        return (hist, bin_centers, bin_edges)

    if use_my_hist:
        hist_ind, centers, edges = my_hist(tie_ind, density=True)
        hist_tot, _, _ = my_hist(jitter,  density=True)
        centers = array(centers)
    else:
        hist_ind, edges = histogram(tie_ind, bins=num_bins, density=True)
        hist_tot, _     = histogram(jitter,  bins=num_bins, density=True)
        centers         = (edges[:-1] + edges[1:]) / 2
    hist_ind_smooth = array(moving_average(hist_ind, n=smooth_width))
    hist_tot_smooth = array(moving_average(hist_tot, n=smooth_width))
    hist_dd = hist_tot_smooth

    # Trying to avoid any residual peak at zero, which can confuse the algorithm:
    center_ix = (num_bins - 1) / 2  # May be fractional.
    peak_ixs  = array(list(filter(lambda x: abs(x - center_ix) > 1,
                                  where(diff(sign(diff(hist_dd))) < 0)[0] + 1)))
    neg_peak_ixs = list(filter(lambda x: x < center_ix, peak_ixs))
    if neg_peak_ixs:
        neg_peak_loc = neg_peak_ixs[argmax(hist_dd[neg_peak_ixs])]
    else:
        neg_peak_loc = int(center_ix)
    pos_peak_ixs = list(filter(lambda x: x > center_ix, peak_ixs))
    if pos_peak_ixs:
        pos_peak_loc = pos_peak_ixs[argmax(hist_dd[pos_peak_ixs])]
    else:
        pos_peak_loc = int(center_ix)
    pjDD = centers[pos_peak_loc] - centers[neg_peak_loc]

    # --- Stash debugging info if an object was provided.
    if dbg_obj:
        dbg_obj.hist_ind_smooth = hist_ind_smooth  # type: ignore
        dbg_obj.centers         = centers  # type: ignore
        dbg_obj.hist_ind        = hist_ind  # type: ignore
        dbg_obj.peak_ixs        = peak_ixs  # type: ignore
        dbg_obj.neg_peak_ixs    = neg_peak_ixs  # type: ignore
        dbg_obj.pos_peak_ixs    = pos_peak_ixs  # type: ignore

    # --- Fit the tails and average the results, to determine Rj.
    pos_max = hist_dd[pos_peak_loc]
    neg_max = hist_dd[neg_peak_loc]
    dd_soltn = [[pos_max, neg_max],]
    pos_tail_ixs = where(hist_dd[pos_peak_loc:] < pos_max / 2)[0]
    neg_tail_ixs = where(hist_dd[:neg_peak_loc] < neg_max / 2)[0]
    if len(pos_tail_ixs) > 0 and len(neg_tail_ixs) > 0:
        pos_tail_ix = pos_tail_ixs[0] + pos_peak_loc  # index of first  piece of right tail
        neg_tail_ix = neg_tail_ixs[-1]                # index of last piece of left tail
        dd_soltn[0].append(pos_tail_ix)
        dd_soltn[0].append(neg_tail_ix)
        # Don't send first or last histogram elements to curve fitter, due to their special nature.
        try:
            popt, pcov = curve_fit(gaus_pdf, centers[pos_tail_ix:-1] * 1e12, hist_dd[pos_tail_ix:-1] * 1e-12)
            mu_pos, sigma_pos = popt
            mu_pos    *= 1e-12  # back to (s)
            sigma_pos *= 1e-12
            err_pos    = sqrt(diag(pcov)) * 1e-12
            dd_soltn  += [[mu_pos, sigma_pos, err_pos],]
        except Exception as err:  # pylint: disable=broad-exception-caught
            mu_pos = 0
            sigma_pos = 0
            dd_soltn += [[err],]
        try:
            popt, pcov = curve_fit(gaus_pdf, centers[1:neg_tail_ix] * 1e12, hist_dd[1:neg_tail_ix] * 1e-12)
            mu_neg, sigma_neg = popt
            mu_neg    *= 1e-12  # back to (s)
            sigma_neg *= 1e-12
            err_neg    = sqrt(diag(pcov)) * 1e-12
            dd_soltn  += [[mu_neg, sigma_neg, err_neg],]
        except Exception as err:  # pylint: disable=broad-exception-caught
            mu_neg = 0
            sigma_neg = 0
            dd_soltn += [[err],]
    else:
        mu_pos = 0
        sigma_pos = 0
        mu_neg = 0
        sigma_neg = 0
    rjDD = (sigma_pos + sigma_neg) / 2
    if dbg_obj:
        dbg_obj.dd_soltn = dd_soltn  # type: ignore

    return (  # pylint: disable=duplicate-code
        jitter,
        array(t_jitter),
        isi,
        dcd,
        float(pj),
        rj,
        pjDD,
        rjDD,
        tie_ind,
        thresh[:half_len],
        jitter_spectrum,
        tie_ind_spectrum,
        jitter_freqs,
        hist_tot_smooth,
        hist_ind_smooth,
        centers,  # Returning just one requires `use_my_hist` True.
        mu_pos,
        mu_neg
    )
