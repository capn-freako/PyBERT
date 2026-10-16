"""
Jitter utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>

Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

from typing import Optional

import numpy as np
from numpy import (
    argmax, array, concatenate, diag, diff, flip,
    histogram, maximum, mean, ones, real, reshape, sign,
    sort, sqrt, tile, where, zeros
)
from numpy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from ..common import Rvec

from .math import gaus_pdf
from .sigproc import moving_average, resize_zero_pad

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
        Array of signal threshold crossing times.
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
        The signal threshold crossing times.
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
    zero_mean: bool = True, dbg_obj: Optional[object] = None, smooth_width: int = 5,
    xing_mask: Optional[Rvec] = None
) -> tuple[Rvec, Rvec, float, float, float, float, float, float,
           Rvec, Rvec, Rvec, Rvec, Rvec, Rvec, Rvec, Rvec, float, float, float, float]:
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
        xing_mask: One flag per ideal crossing, marking those that carry timing information.
            ISI, DCD, and both jitter histograms (and so the dual-Dirac fit and bathtub built from them)
            use only these; the rest still take part in the TIE track, its rising/falling pairing and
            pattern averaging, and the jitter spectra. (e.g. - For PAM-4, only transitions symmetric about
            the 0V threshold cross it mid-UI; the others cross early or late, by an amount set by the data
            and the edge shape, not by the timing.)
            Default: None (all crossings)

    Returns:
        A tuple containing

            - The total jitter.
            - The times (taken from 'ideal_xings') corresponding to the returned jitter values.
            - The peak to peak jitter due to intersymbol interference (ISI).
            - The peak to peak jitter due to duty cycle distortion (DCD).
            - The peak to peak jitter due to uncorrelated periodic sources (Pj).
            - The standard deviation of the jitter due to uncorrelated unbounded random sources (Rj).
            - Dual-Dirac peak to peak jitter.
            - Dual-Dirac random jitter.
            - The data independent jitter.
            - Threshold for determining periodic components.
            - The spectral magnitude of the total jitter.
            - The spectral magnitude of the data independent jitter.
            - The frequencies corresponding to the spectrum components.
            - The smoothed histogram of the total jitter.
            - The smoothed histogram of the data-independent jitter.
            - The bin center values for both histograms.
            - The mean of the Gaussian distribution best fitted to the right tail.
            - The mean of the Gaussian distribution best fitted to the left tail.
            - The latest time (relative to ideal) of the bounded (data dependent + periodic) jitter.
            - The earliest time (relative to ideal) of the bounded (data dependent + periodic) jitter.
            (The dual-Dirac values -- pjDD, rjDD, and the fitted tail means -- are for reference only;
            for extrapolating jitter tails, use Rj's Gaussian, starting from the bounded jitter's extent.)

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

    # Assemble the TIE track: one entry per ideal crossing, each matched on its own to the actual
    # crossings within [-UI/2, +UI/2] of it (averaged, since noise may produce several).
    i = 0
    jitterL = []
    t_jitter = []
    foundL = []
    for ideal_xing in ideal_xings:
        min_t = ideal_xing - ui / 2.0
        max_t = ideal_xing + ui / 2.0
        while i < len(actual_xings) and actual_xings[i] < min_t:
            i += 1
        if i == len(actual_xings):  # We've exhausted the list of actual crossings; we're done.
            break
        xings = []
        j = i
        while j < len(actual_xings) and actual_xings[j] <= max_t:
            xings.append(actual_xings[j])
            j += 1
        if xings:
            jitterL.append(mean(xings) - ideal_xing)
            foundL.append(True)
        else:  # It didn't cross within [-UI/2, +UI/2]: +/-3UI/4 lands in the histogram's end bins.
            jitterL.append(3.0 * ui / 4.0 if len(jitterL) % 2 == 0 else -3.0 * ui / 4.0)
            foundL.append(False)
        t_jitter.append(ideal_xing)
    jitter = array(jitterL)
    found = array(foundL, dtype=bool)
    if xing_mask is None:
        mask = ones(len(jitter), dtype=bool)
    else:
        mask = array(xing_mask, dtype=bool)[:len(jitter)]
    measured = mask & found  # informative crossings with a real measured TIE

    # ToDo: Report this in the status bar.
    if len(jitter) == 0 or len(t_jitter) == 0:
        print("No crossings found!", flush=True)

    if debug:
        print("mean(jitter):", mean(jitter))
        print("len(jitter):", len(jitter))

    if zero_mean and measured.any():
        jitter[found] -= mean(jitter[measured])

    # Do the jitter decomposition.
    # - Use averaging over the pattern period to remove the uncorrelated components,
    #   before calculating data dependent components (from measured crossings only).
    def pattern_ave(x: Rvec, ok: Rvec, n_per: int) -> tuple[Rvec, Rvec]:
        "Average `x` over the pattern repetitions, using only `ok` entries; also count them, per position."
        _x = reshape(resize_zero_pad(where(ok, x, 0.0), num_patterns * n_per), (num_patterns, n_per))
        _n = reshape(resize_zero_pad(ok.astype(float), num_patterns * n_per), (num_patterns, n_per)).sum(axis=0)
        return _x.sum(axis=0) / maximum(_n, 1), _n

    # - Separate the rising and falling edges.
    tie_risings_ave, risings_ok = pattern_ave(jitter[0::2], found[0::2], xings_per_pattern // 2)
    tie_fallings_ave, fallings_ok = pattern_ave(jitter[1::2], found[1::2], xings_per_pattern // 2)
    # (The ideal crossings repeat with the pattern, so the first pattern's flags hold for all.)
    risings_ave  = tie_risings_ave[mask[0:xings_per_pattern:2] & (risings_ok > 0)]
    fallings_ave = tie_fallings_ave[mask[1:xings_per_pattern:2] & (fallings_ok > 0)]
    isi = float(max(np.ptp(risings_ave) if len(risings_ave) else 0.0,
                    np.ptp(fallings_ave) if len(fallings_ave) else 0.0))
    isi = min(isi, ui)  # Cap the ISI at the unit interval.
    dcd = abs(mean(risings_ave) - mean(fallings_ave)) if len(risings_ave) and len(fallings_ave) else 0.0

    # - Subtract the data dependent jitter (the pattern average, repeated for every pattern)
    #   from the original TIE track, in order to yield the data independent jitter.
    ave, n_ave = pattern_ave(jitter, found, xings_per_pattern)
    n_reps = int(len(jitter) // xings_per_pattern + 1)
    tie_ave = tile(ave, n_reps)[:len(jitter)]
    n_xing = tile(n_ave, n_reps)[:len(jitter)]
    # A pattern average over `n` repetitions includes 1/n of each one's own data independent jitter,
    # leaving only (1 - 1/n) of its variance in the difference; undo that. With `n` = 1, nothing is left:
    # such crossings carry no data independent information, and are left out of its statistics.
    usable = measured & (n_xing >= 2)
    tie_ind = (jitter - tie_ave) * sqrt(n_xing / maximum(n_xing - 1, 1))
    if zero_mean and usable.any():
        tie_ind -= mean(tie_ind[usable])

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
        spl = interp1d(t_jitter, where(found, jitter, 0.0), bounds_error=False, fill_value="extrapolate")
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
        spl = interp1d(array(t_jitter)[usable], tie_ind[usable], bounds_error=False, fill_value="extrapolate")
        tie_ind_interp = spl(t)
        y = fft(tie_ind_interp)
        y_mag = abs(y)
        tie_ind_spectrum = y_mag[:half_len]
    except Exception as err:  # pylint: disable=broad-exception-caught
        print(f"t_jitter: {t_jitter}")
        print(f"tie_ind: {tie_ind}")
        print(f"Error calculating data independent TIE: {err}", flush=True)
        y = zeros(len(t))  # type: ignore
        y_mag = zeros(len(t))
        tie_ind_spectrum = zeros(half_len)

    # -- Perform spectral extraction of Pj from the data independent jitter,
    # -- using a threshold based on a moving average to identify the periodic components.
    win_width = 100
    y_mean    = moving_average(y_mag,                 n=win_width)
    y_var     = moving_average((y_mag - y_mean) ** 2, n=win_width)
    y_sigma   = sqrt(y_var)
    thresh    = y_mean + rel_thresh * y_sigma
    y_per     = where(y_mag > thresh, y, zeros(len(y)))  # Periodic components are those lying above the threshold.
    tie_per   = real(ifft(y_per))
    pj        = np.ptp(tie_per)
    # Rj is measured at the crossings themselves: the uniformly resampled track used for the spectrum
    # interpolates between them, which understates the variance of what's left once Pj is removed.
    # (No usable crossings -- e.g. fewer than two pattern repetitions -- leaves no data independent information.)
    tie_per_x = np.interp(array(t_jitter)[usable], t, tie_per) if usable.any() else zeros(0)
    tie_rnd_x = tie_ind[usable] - tie_per_x
    rj        = float(tie_rnd_x.std()) if len(tie_rnd_x) else 0.0
    # The bounded (data dependent + periodic) jitter's extent: where Rj's Gaussian tails start.
    bounded   = tie_ave[usable] + tie_per_x
    dj_pos    = float(bounded.max()) if len(bounded) else 0.0
    dj_neg    = float(bounded.min()) if len(bounded) else 0.0

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
        hist_ind, centers, edges = my_hist(tie_ind[usable], density=True)
        hist_tot, _, _ = my_hist(jitter[mask],  density=True)
        centers = array(centers)
    else:
        hist_ind, edges = histogram(tie_ind[usable], bins=num_bins, density=True)
        hist_tot, _     = histogram(jitter[mask],  bins=num_bins, density=True)
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
        # Fit in (ps), and densities per (ps): in (s), `curve_fit()` can't move from its default starting point.
        # Start each fit from its peak location, and the peak-to-half-maximum distance (~1.2 sigma).
        x_ps = centers * 1e12
        y_ps = hist_dd * 1e-12
        dx_ps = x_ps[2] - x_ps[1]
        # Don't send first or last histogram elements to curve fitter, due to their special nature.
        try:
            popt, pcov = curve_fit(gaus_pdf, x_ps[pos_tail_ix:-1], y_ps[pos_tail_ix:-1],
                                   p0=(x_ps[pos_peak_loc], max(x_ps[pos_tail_ix] - x_ps[pos_peak_loc], dx_ps)))
            mu_pos, sigma_pos = popt[0], abs(popt[1])
            mu_pos    *= 1e-12  # back to (s)
            sigma_pos *= 1e-12
            err_pos    = sqrt(diag(pcov)) * 1e-12
            dd_soltn  += [[mu_pos, sigma_pos, err_pos],]
        except Exception as err:  # pylint: disable=broad-exception-caught
            mu_pos = 0
            sigma_pos = 0
            dd_soltn += [[err],]
        try:
            popt, pcov = curve_fit(gaus_pdf, x_ps[1: neg_tail_ix + 1], y_ps[1: neg_tail_ix + 1],
                                   p0=(x_ps[neg_peak_loc], max(x_ps[neg_peak_loc] - x_ps[neg_tail_ix], dx_ps)))
            mu_neg, sigma_neg = popt[0], abs(popt[1])
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
        mu_neg,
        dj_pos,
        dj_neg,
    )
