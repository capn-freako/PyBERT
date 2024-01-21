"""General purpose utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 27, 2014 (Copied from pybert_cntrl.py.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
import importlib
import os.path
import pkgutil
import re
import numpy as np
import skrf  as rf
from cmath     import phase, rect
from functools import reduce
from numpy     import (
    append, argmax, array, concatenate, convolve, cumsum, diff, exp,
    histogram, insert, linspace, log, log10, maximum, mean, ones,
    pad, pi, power, real, reshape, resize, roll, sign,
    sort, sqrt, where, zeros,
)
from numpy.fft         import fft, ifft, fftshift, irfft, rfft
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.linalg      import inv
from scipy.optimize    import curve_fit, minimize
from scipy.signal      import freqs, invres
from scipy.stats       import norm

debug          = False
gDebugOptimize = False
gMaxCTLEPeak   = 20  # max. allowed CTLE peaking (dB) (when optimizing, only)


def moving_average(a, n=3):
    """Calculates a sliding average over the input vector.

    Uses a weighted averaging kernel, to preserve singularity
    of peak location in input data.

    Args:
        a([float]): Input vector to be averaged.
        n(int): Width of averaging window, in vector samples.
            Odd numbers work best.
            (Optional; default = 3.)

    Returns:
        [float]: the moving average of the input vector, leaving the input
            vector unchanged.

    Notes:
        1. The odd code is intended to "protect" the first/last elements
           of the input vector from the averaging process.
           In PyBERT those elements "collect" the missed edges when
           assembling the TIE.
           Because of this non-standard use, those bins shouldn't be
           included in averaging.
    """
    rect = ones((n+1)//2)
    krnl = convolve(rect, rect)
    krnl = krnl / krnl.sum()
    res  = convolve(a[1:-1], krnl, mode='same')
    return array([a[0]] + list(res) + [a[-1]])


def find_crossing_times(
    t,
    x,
    min_delay: float = 0.0,
    rising_first: bool = True,
    min_init_dev: float = 0.1,
    thresh: float = 0.0,
):
    """Finds the threshold crossing times of the input signal.

    Args:
        t([float]): Vector of sample times. Intervals do NOT need to be
            uniform.
        x([float]): Sampled input vector.
        min_delay(float): Minimum delay required, before allowing
            crossings. (Helps avoid false crossings at beginning of
            signal.) (Optional; default = 0.)
        rising_first(bool): When True, start with the first rising edge
            found. (Optional; default = True.) When this option is True,
            the first rising edge crossing is the first crossing returned.
            This is the desired behavior for PyBERT, because we always
            initialize the bit stream with [0, 0, 1, 1], in order to
            provide a known synchronization point for jitter analysis.
        min_init_dev(float): The minimum initial deviation from zero,
            which must be detected, before searching for crossings.
            Normalized to maximum input signal magnitude.
            (Optional; default = 0.1.)
        thresh(float): Vertical crossing threshold.

    Returns:
        [float]: Array of signal threshold crossing times.
    """

    if len(t) != len(x):
        raise ValueError(f"len(t) ({len(t)}) and len(x) ({len(x)}) need to be the same.")

    t = array(t)
    x = array(x)

    try:
        max_mag_x = max(abs(x))
    except:
        print("len(x):", len(x))
        raise
    min_mag_x = min_init_dev * max_mag_x
    i = 0
    while abs(x[i]) < min_mag_x:
        i += 1
        assert i < len(x), "Input signal minimum deviation not detected!"
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
        assert min_delay < xings[-1], f"min_delay ({min_delay}) must be less than last crossing time ({xings[-1]})."
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
    except:
        print("len(diff_sign_x):", len(diff_sign_x))
        print("len(xing_ix):", len(xing_ix))
        print("i:", i)
        raise

    return array(xings[i:])


def find_crossings(
    t,
    x,
    amplitude=1.0,
    min_delay: float = 0.0,
    rising_first: bool = True,
    min_init_dev=0.1,
    mod_type=0,
):
    """Finds the crossing times in a signal, according to the modulation type.

    Args:
        t([float]): The times associated with each signal sample.
        x([float]): The signal samples.
        amplitude(float): The nominal signal amplitude. (Used for
            determining thresholds, in the case of some modulation
            types.)
        min_delay(float): The earliest possible sample time we want
            returned. (Optional; default = 0.)
        rising_first(bool): When True, start with the first rising edge
            found. When this option is True, the first rising edge
            crossing is the first crossing returned. This is the desired
            behavior for PyBERT, because we always initialize the bit
            stream with [0, 0, 1, 1], in order to provide a known
            synchronization point for jitter analysis.
            (Optional; default = True.)
        min_init_dev(float): The minimum initial deviation from zero,
            which must be detected, before searching for crossings.
            Normalized to maximum input signal magnitude.
            (Optional; default = 0.1.)
        mod_type(int): The modulation type. Allowed values are:
            {0: NRZ, 1: Duo-binary, 2: PAM-4}
            (Optional; default = 0.)

    Returns:
        [float]: The signal threshold crossing times.
    """

    assert mod_type >= 0 and mod_type <= 2, f"ERROR: pybert_util.find_crossings(): Unknown modulation type: {mod_type}"

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


def gaus_pdf(x, mu, sigma):
    """
    Gaussian probability density function.
    """
    sqrt_2pi = np.sqrt(2 * np.pi)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt_2pi)


def calc_jitter(
    ui, nui, pattern_len, ideal_xings, actual_xings,
    rel_thresh=3.0, num_bins=100, zero_mean=True, dbg_obj=None, smooth_width=5
):
    """
    Calculate the jitter in a set of actual zero crossings,
    given the ideal crossings and unit interval.

    Args:
        ui(float): The nominal unit interval.
        nui(int): The number of unit intervals spanned by the input signal.
        pattern_len(int): The number of unit intervals, before input symbol stream repeats.
        ideal_xings([float]): The ideal zero crossing locations of the edges.
        actual_xings([float]): The actual zero crossing locations of the edges.

    KeywordArgs:
        rel_thresh(float): The threshold for determining periodic jitter spectral components (sigma).
            (Default: 3.0)
        num_bins(int): The number of bins to use, when forming histograms.
            (Default: 99)
        zero_mean(bool): Force the mean jitter to zero, when True.
            (Default: True)
        dbg_obj(object): Object for stashing debugging info.
            (Default: None)
        smooth_width(int): Width of smoothing window to use when calculating moving averages.
            (Default: 5)

    Returns:
        ( [real]: The total jitter.
        , [real]: The times (taken from 'ideal_xings') corresponding to the returned jitter values.
        , real: The peak to peak jitter due to intersymbol interference (ISI).
        , real: The peak to peak jitter due to duty cycle distortion (DCD).
        , real: The peak to peak jitter due to uncorrelated periodic sources (Pj).
        , real: The standard deviation of the jitter due to uncorrelated unbounded random sources (Rj).
        , [real]: The data independent jitter.
        , [real]: Threshold for determining periodic components.
        , [real]: The spectral magnitude of the total jitter.
        , [real]: The spectral magnitude of the data independent jitter.
        , [real]: The frequencies corresponding to the spectrum components.
        , [real]: The smoothed histogram of the total jitter.
        , [real]: The smoothed histogram of the data-independent jitter.
        , [real]: The bin center values for both histograms.
        )

    Raises:
        ValueError: If input checking fails, or curve fitting goes awry.
        AssertionError: If less than one full pattern given as input, or an odd number of crossings per pattern was detected.

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
    assert num_patterns, f"Need at least one full pattern repetition! (pattern_len: {pattern_len}; nui: {nui})"
    xings_per_pattern = where(ideal_xings > (pattern_len * ui))[0][0]
    if xings_per_pattern % 2 or not xings_per_pattern:
        print("xings_per_pattern:", xings_per_pattern)
        print("min(ideal_xings):", min(ideal_xings))
        raise AssertionError("pybert_util.calc_jitter(): Odd number of (or, no) crossings per pattern detected!")

    # Assemble the TIE track.
    i = 0
    jitter = []
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
            jitter.append( 3.0 * ui / 4.0)  # Pad the jitter w/ alternating +/- 3UI/4.
            jitter.append(-3.0 * ui / 4.0)  # (Will get pulled into [-UI/2, UI/2], later.
            skip_next_ideal_xing = True  # If we missed one, we missed two.
        else:  # Noise may produce several crossings.
            xings = []  # We find all those within the interval [-UI/2, +UI/2]
            j = i  # centered around the ideal crossing, and take the average.
            while j < len(actual_xings) and actual_xings[j] <= max_t:
                xings.append(actual_xings[j])
                j += 1
            tie = mean(xings) - ideal_xing
            jitter.append(tie)
        t_jitter.append(ideal_xing)
    jitter = array(jitter)

    if debug:
        print("mean(jitter):", mean(jitter))
        print("len(jitter):", len(jitter))

    if zero_mean:
        jitter -= mean(jitter)

    # Do the jitter decomposition.
    # - Separate the rising and falling edges, shaped appropriately for averaging over the pattern period.
    tie_risings  = jitter.take(list(range(0, len(jitter), 2)))
    tie_fallings = jitter.take(list(range(1, len(jitter), 2)))
    tie_risings.resize(num_patterns * xings_per_pattern // 2, refcheck=False)
    tie_fallings.resize(num_patterns * xings_per_pattern // 2, refcheck=False)
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
    _jitter.resize(num_patterns * xings_per_pattern)
    tie_ave = resize(reshape(_jitter, (num_patterns, xings_per_pattern)).mean(axis=0), len(jitter))
    tie_ind = jitter - tie_ave
    if zero_mean:
        tie_ind -= mean(tie_ind)

    # - Calculate the total and data-independent jitter spectrums, for display purposes only.
    # -- Calculate the relevant time/frequency vectors.
    osf = 1                                             # jitter oversampling factor
    t0  = ui / osf                                      # jitter sampling period
    t   = np.array([n * t0 for n in range(nui * osf)])  # jitter samples time vector
    f0  = 1.0 / (ui * nui)                              # jitter samples fundamental frequency
    f   = [n * f0 for n in range(len(t) // 2)]          # [0:f0:fNyquist)
    f   = np.array(f + [1 / (2 * t0)] + list(-1 * np.flip(np.array(f[1:]))))  # [0:f0:fN) ++ [fN:-f0:0)
    half_len = len(f) // 2                              # for spectrum plotting convenience

    # -- Make TIE vector uniformly sampled in time, via interpolation, for use as input to `fft()`.
    # spl = UnivariateSpline(t_jitter, jitter)  # Way of the future, but does funny things. :(
    spl = interp1d(t_jitter, jitter, bounds_error=False, fill_value="extrapolate")
    tie_interp = spl(t)
    y = fft(tie_interp)
    jitter_spectrum = abs(y[:half_len])
    jitter_freqs    = f[:half_len]

    # -- Repeat for data-independent jitter.
    spl = interp1d(t_jitter, tie_ind, bounds_error=False, fill_value="extrapolate")
    tie_ind_interp = spl(t)
    y = fft(tie_ind_interp)
    y_mag = abs(y)
    tie_ind_spectrum = y_mag[:half_len]

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
    pj        = tie_per.ptp()
    tie_rnd   = real(ifft(y_rnd))
    rj        = np.sqrt(np.mean((tie_rnd - tie_rnd.mean())**2))

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
        bin_centers = [-ui/2] + list((bin_edges[1:-2] + bin_edges[2:-1]) / 2) + [ui/2]
        hist, _     = histogram(x, bin_edges)
        hist        = hist / hist.sum()  # PMF
        if density:
            hist /= diff(bin_edges)
        return (hist, bin_centers)

    if use_my_hist:
        hist_ind, centers = my_hist(tie_ind, density=True)
        hist_tot, _       = my_hist(jitter,  density=True)
        centers           = array(centers)
    else:
        hist_ind, edges = histogram(tie_ind, bins=num_bins, density=True)
        hist_tot, _     = histogram(jitter,  bins=num_bins, density=True)
        centers         = (edges[:-1] + edges[1:]) / 2
    hist_ind_smooth = array(moving_average(hist_ind, n=smooth_width))
    hist_tot_smooth = array(moving_average(hist_tot, n=smooth_width))
    hist_dd = hist_tot_smooth
    # Trying to avoid any residual peak at zero, which can confuse the algorithm:
    center_ix = (num_bins-1)/2  # May be fractional.
    peak_ixs  = array(list(filter( lambda x: abs(x - center_ix) > 1
                                , where(diff(sign(diff(hist_dd))) < 0)[0] + 1 )))
    neg_peak_ixs = list(filter(lambda x: x < center_ix, peak_ixs))
    if len(neg_peak_ixs):
        neg_peak_loc = neg_peak_ixs[argmax(hist_dd[neg_peak_ixs])]
    else:
        neg_peak_loc = int(center_ix)
    pos_peak_ixs = list(filter(lambda x: x > center_ix, peak_ixs))
    if len(pos_peak_ixs):
        pos_peak_loc = pos_peak_ixs[argmax(hist_dd[pos_peak_ixs])]
    else:
        pos_peak_loc = int(center_ix)
    pjDD = (centers[pos_peak_loc] - centers[neg_peak_loc])

    # --- Stash debugging info if an object was provided.
    if dbg_obj:
        dbg_obj.hist_ind_smooth = hist_ind_smooth
        dbg_obj.centers         = centers
        dbg_obj.hist_ind        = hist_ind
        dbg_obj.peak_ixs        = peak_ixs
        dbg_obj.neg_peak_ixs    = neg_peak_ixs
        dbg_obj.pos_peak_ixs    = pos_peak_ixs

    # --- Fit the tails and average the results, to determine Rj.
    pos_max     = hist_dd[pos_peak_loc]
    neg_max     = hist_dd[neg_peak_loc]
    pos_tail_ix = where(hist_dd[pos_peak_loc:] < pos_max / 2)[0] + pos_peak_loc
    neg_tail_ix = where(hist_dd[:neg_peak_loc] < neg_max / 2)[0]
    dd_soltn    = []
    try:
        popt, pcov = curve_fit(gaus_pdf, centers[pos_tail_ix]*1e12, hist_dd[pos_tail_ix]*1e-12)
        mu_pos, sigma_pos = popt
        mu_pos    *= 1e-12  # back to (s)
        sigma_pos *= 1e-12
        err_pos    = np.sqrt(np.diag(pcov)) * 1e-12
        dd_soltn   = [mu_pos, sigma_pos, err_pos]
    except:
        sigma_pos = 0
    try:
        popt, pcov = curve_fit(gaus_pdf, centers[neg_tail_ix]*1e12, hist_dd[neg_tail_ix]*1e-12)
        mu_neg, sigma_neg = popt
        mu_neg    *= 1e-12  # back to (s)
        sigma_neg *= 1e-12
        err_neg    = np.sqrt(np.diag(pcov)) * 1e-12
        dd_soltn  += [mu_neg, sigma_neg, err_neg]
    except:
        sigma_neg = 0
    rjDD = (sigma_pos + sigma_neg) / 2
    if dbg_obj:
        dbg_obj.dd_soltn = dd_soltn

    return (
        jitter,
        t_jitter,
        isi,
        dcd,
        pj,
        rj,
        pjDD,
        rjDD,
        tie_ind,
        thresh[:half_len],
        jitter_spectrum,
        tie_ind_spectrum,
        jitter_freqs,
        # hist_tot,
        hist_tot_smooth,
        # hist_ind,
        hist_ind_smooth,
        centers,  # Returning just one requires `use_my_hist` True.
    )


def make_uniform(t, jitter, ui, nbits):
    """Make the jitter vector uniformly sampled in time, by zero-filling where
    necessary.

    The trick, here, is creating a uniformly sampled input vector for the FFT operation,
    since the jitter samples are almost certainly not uniformly sampled.
    We do this by simply zero padding the missing samples.

    Inputs:

    - t      : The sample times for the 'jitter' vector.

    - jitter : The input jitter samples.

    - ui     : The nominal unit interval.

    - nbits  : The desired number of unit intervals, in the time domain.

    Output:

    - y      : The uniformly sampled, zero padded jitter vector.

    - y_ix   : The indices where y is valid (i.e. - not zero padded).
    """

    if len(t) < len(jitter):
        jitter = jitter[: len(t)]

    run_lengths = list(map(int, diff(t) / ui + 0.5))
    valid_ix = [0] + list(cumsum(run_lengths))
    valid_ix = [x for x in valid_ix if x < nbits]
    missing = where(array(run_lengths) > 1)[0]
    num_insertions = 0
    jitter = list(jitter)  # Because we use 'insert'.

    for i in missing:
        for _ in range(run_lengths[i] - 1):
            jitter.insert(i + 1 + num_insertions, 0.0)
            num_insertions += 1

    if len(jitter) < nbits:
        jitter.extend([0.0] * (nbits - len(jitter)))
    if len(jitter) > nbits:
        jitter = jitter[:nbits]

    return jitter, valid_ix


def calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, ws):
    """Calculates propagation constant from cross-sectional parameters.

    The formula's applied are taken from Howard Johnson's "Metallic Transmission Model"
    (See "High Speed Signal Propagation", Sec. 3.1.)

    Inputs:
      - R0          skin effect resistance (Ohms/m)
      - w0          cross-over freq.
      - Rdc         d.c. resistance (Ohms/m)
      - Z0          characteristic impedance in LC region (Ohms)
      - v0          propagation velocity (m/s)
      - Theta0      loss tangent
      - ws          frequency sample points vector

    Outputs:
      - gamma       frequency dependent propagation constant
      - Zc          frequency dependent characteristic impedance
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    Rac = R0 * sqrt(2 * 1j * w / w0)  # AC resistance vector
    R = sqrt(power(Rdc, 2) + power(Rac, 2))  # total resistance vector
    L0 = Z0 / v0  # "external" inductance per unit length (H/m)
    C0 = 1.0 / (Z0 * v0)  # nominal capacitance per unit length (F/m)
    C = C0 * power((1j * w / w0), (-2.0 * Theta0 / pi))  # complex capacitance per unit length (F/m)
    gamma = sqrt((1j * w * L0 + R) * (1j * w * C))  # propagation constant (nepers/m)
    Zc = sqrt((1j * w * L0 + R) / (1j * w * C))  # characteristic impedance (Ohms)
    Zc[0] = Z0  # d.c. impedance blows up and requires correcting.

    return (gamma, Zc)


def calc_gamma_RLGC(R, L, G, C, ws):
    """Calculates propagation constant from R, L, G, and C.

    Inputs:
      - R           resistance per unit length (Ohms/m)
      - L           inductance per unit length (Henrys/m)
      - G           conductance per unit length (Siemens/m)
      - C           capacitance per unit length (Farads/m)
      - ws          frequency sample points vector

    Outputs:
      - gamma       frequency dependent propagation constant
      - Zc          frequency dependent characteristic impedance
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    gamma = sqrt((1j * w * L0 + R) * (1j * w * C + G))  # propagation constant (nepers/m)
    Zc = sqrt((1j * w * L0 + R) / (1j * w * C + G))  # characteristic impedance (Ohms)

    return (gamma, Zc)


def calc_G(H, Rs, Cs, Zc, RL, Cp, ws):
    """Calculates fully loaded transfer function of complete channel.

    Inputs:
      - H     unloaded transfer function of interconnect
      - Rs    source series resistance (differential)
      - Cs    source parallel (parasitic) capacitance (single ended)
      - Zc    frequency dependent characteristic impedance of the interconnect
      - RL    load resistance (differential)
      - Cp    load parallel (parasitic) capacitance (single ended)
      - ws    frequency sample points vector

    Outputs:
      - G     transfer function of fully loaded channel
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12
    if Cp == 0:
        Cp = 1e-18

    def Rpar2C(R, C):
        """Calculates the impedance of the parallel combination of `R` with two
        `C`s in series."""
        return R / (1.0 + 1j * w * R * C / 2)

    # Impedance looking back into the Tx output is a simple parallel RC network.
    Zs = Rpar2C(Rs, Cs)  # The parasitic capacitances are in series.
    # Rx load impedance is parallel comb. of Rterm & parasitic cap.
    # (The two parasitic capacitances are in series.)
    ZL = Rpar2C(RL, Cp)
    # Admittance into the interconnect is (Cs || Zc) / (Rs + (Cs || Zc)).
    Cs_par_Zc = Rpar2C(Zc, Cs)
    Y = Cs_par_Zc / (Rs + Cs_par_Zc)
    # Reflection coefficient at Rx:
    R1 = (ZL - Zc) / (ZL + Zc)
    # Reflection coefficient at Tx:
    R2 = (Zs - Zc) / (Zs + Zc)
    # Fully loaded channel transfer function:
    return Y * H * (1 + R1) / (1 - R1 * R2 * H**2)


def calc_eye(ui, samps_per_ui, height, ys, y_max, clock_times=None):
    """Calculates the "eye" diagram of the input signal vector.

    Args:
        ui(float): unit interval (s)
        samps_per_ui(int): # of samples per unit interval
        height(int): height of output image data array
        ys([float]): signal vector of interest
        y_max(float): max. +/- vertical extremity of plot

    Keyword Args:
        clock_times([float]): (optional) vector of clock times to use
            for eye centers. If not provided, just use mean
            zero-crossing and assume constant UI and no phase jumps.
            (This allows the same function to be used for eye diagram
            creation, for both pre and post-CDR signals.)

    Returns:
        2D *NumPy* array: The "heat map" representing the eye diagram. Each grid
            location contains a value indicating the number of times the
            signal passed through that location.
    """

    # List/array necessities.
    ys = array(ys)

    # Intermediate variable calculation.
    tsamp = ui / samps_per_ui

    # Adjust the scaling.
    width = 2 * samps_per_ui
    y_scale = height // (2 * y_max)  # (pixels/V)
    y_offset = height // 2  # (pixels)

    # Generate the "heat" picture array.
    img_array = zeros([height, width])
    if clock_times:
        for clock_time in clock_times:
            start_time = clock_time - ui
            start_ix = int(start_time / tsamp)
            if start_ix + 2 * samps_per_ui > len(ys):
                break
            interp_fac = (start_time - start_ix * tsamp) // tsamp
            i = 0
            for samp1, samp2 in zip(
                ys[start_ix : start_ix + 2 * samps_per_ui],
                ys[start_ix + 1 : start_ix + 1 + 2 * samps_per_ui],
            ):
                y = samp1 + (samp2 - samp1) * interp_fac
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
                i += 1
    else:
        start_ix = where(diff(sign(ys)))[0][0] + samps_per_ui // 2
        last_start_ix = len(ys) - 2 * samps_per_ui
        while start_ix < last_start_ix:
            i = 0
            for y in ys[start_ix : start_ix + 2 * samps_per_ui]:
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
                i += 1
            start_ix += samps_per_ui

    return img_array


def make_ctle(rx_bw, peak_freq, peak_mag, w, mode="Passive", dc_offset=0):
    """Generate the frequency response of a continuous time linear equalizer
    (CTLE), given the:

    - signal path bandwidth,
    - peaking specification
    - list of frequencies of interest, and
    - operational mode/offset.

    We use the 'invres()' function from scipy.signal, as it suggests
    itself as a natural approach, given our chosen use model of having
    the user provide the peaking frequency and degree of peaking.

    That is, we define our desired frequency response using one zero
    and two poles, where:

    - The pole locations are equal to:
       - the signal path natural bandwidth, and
       - the user specified peaking frequency.

    - The zero location is chosen, so as to provide the desired degree
      of peaking.

    Inputs:

      - rx_bw        The natural (or, unequalized) signal path bandwidth (Hz).

      - peak_freq    The location of the desired peak in the frequency
                     response (Hz).

      - peak_mag     The desired relative magnitude of the peak (dB). (mag(H(0)) = 1)

      - w            The list of frequencies of interest (rads./s).

      - mode         The operational mode; must be one of:
                       - 'Off'    : CTLE is disengaged.
                       - 'Passive': Maximum frequency response has magnitude one.
                       - 'AGC'    : Automatic gain control. (Handled by calling routine.)
                       - 'Manual' : D.C. offset is set manually.

      - dc_offset    The d.c. offset of the CTLE gain curve (dB).
                     (Only valid, when 'mode' = 'Manual'.)

    Outputs:

      - w, H         The resultant complex frequency response, at the
                     given frequencies.
    """

    if mode == "Off":
        return (w, ones(len(w)))

    p2 = -2.0 * pi * rx_bw
    p1 = -2.0 * pi * peak_freq
    z = p1 / pow(10.0, peak_mag / 20.0)
    if p2 != p1:
        r1 = (z - p1) / (p2 - p1)
        r2 = 1 - r1
    else:
        r1 = -1.0
        r2 = z - p1
    b, a = invres([r1, r2], [p1, p2], [])
    w, H = freqs(b, a, w)

    if mode == "Passive":
        H /= max(abs(H))
    elif mode in ("Manual", "AGC"):
        H *= pow(10.0, dc_offset / 20.0) / abs(H[0])  # Enforce d.c. offset.
    else:
        raise RuntimeError(f"pybert_util.make_ctle(): Unrecognized value for 'mode' parameter: {mode}.")

    return (w, H)


def make_ctle_gen(z, p1, p2, fHP):
    """Make a CTLE generator, from two poles, one zero, and a low-frequency pole/zero pair.

    Args:
        z(complex): Zero of the frequency response (Hz).
        p1(complex): First pole of the frequency response (Hz).
        p2(complex): Second pole of the frequency response (Hz).
        fHP(real): Frequency of 2nd stage pole/zero  (Hz).

    Returns:
        function: The returned function takes the d.c. gains and angular frequencies of interest
            as arguments, and returns the frequency response.
    """
    A = -2*pi*p1*p2/z

    def ctle_gen(gdc, ghp, w):
        """CTLE generator, made at runtime.

        Args:
            gdc(real): D.c. response.
            ghp(real): D.c. response (low-f).
            w([real]): Angular frequencies of interest (rad./s).

        Returns:
            [complex]: CTLE frequency response at `w`.
        """
        s = 1j*w
        return A*(s - gdc*2*pi*z)*(s - ghp*2*pi*fHP) / ((s-2*pi*p1)*(s-2*pi*p2)*(s-2*pi*fHP))

    return ctle_gen


def trim_impulse(g, min_len=0, max_len=1000000):
    """Trim impulse response, for more useful display, by:

      - clipping off the tail, after 99.8% of the total power has been
        captured (Using 99.9% was causing problems; I don't know why.), and
      - setting the "front porch" length equal to 20% of the total length.

    Inputs:

      - g         impulse response

      - min_len   (optional) minimum length of returned vector

      - max_len   (optional) maximum length of returned vector

    Outputs:

      - g_trim    trimmed impulse response

      - start_ix  index of first returned sample
    """

    # Trim off potential FFT artifacts from the end and capture peak location.
    g = array(g[: int(0.9 * len(g))])
    max_ix = np.argmax(g)

    # Capture 99.8% of the total energy.
    Pt = 0.998 * sum(g**2)
    i = 0
    P = 0
    while P < Pt:
        P += g[i] ** 2
        i += 1
    stop_ix = min(max_ix + max_len, max(i, max_ix + min_len))

    # Set "front/back porch" to 20%, doing appropriate bounds checking.
    length = stop_ix - max_ix
    porch = length // 3
    start_ix = max(0, max_ix - porch)
    stop_ix = min(len(g), stop_ix + porch)
    return (g[start_ix:stop_ix].copy(), start_ix)


def H_2_s2p(H, Zc, fs, Zref=50):
    """Convert transfer function to 2-port network.

    Args:
        H([complex]): transfer function of medium alone.
        Zc([complex]): complex impedance of medium.
        fs([real]): frequencies at which `H` and `Zc` were sampled (Hz).

    KeywordArgs:
        Zref(real): reference (i.e. - port) impedance to be used in constructing the network (Ohms). (Default: 50)

    Returns:
        skrf.Network: 2-port network representing the channel to which `H` and `Zc` pertain.
    """
    # ToDo: Fix this code.
    ws = 2 * pi * fs
    G = calc_G(H, Zref, 0, Zc, Zref, 0, ws)  # See `calc_G()` docstring.
    R1 = (Zc - Zref) / (Zc + Zref)  # reflection coefficient looking into medium from port
    T1 = 1 + R1  # transmission coefficient looking into medium from port
    # Z2   = Zc * (1 - R1*H**2)         # impedance looking into port 2, with port 1 terminated into Zref
    # R2   = (Z2 - Zc) / (Z2 + Zc)      # reflection coefficient looking out of port 2
    # R2   = 0
    # Z1   = Zc * (1 + R2*H**2)         # impedance looking into port 1, with port 2 terminated into Z2
    # Calculate the one-way transfer function of medium capped w/ ports of the chosen impedance.
    # G    = calc_G(H, Zref, 0, Zc, Zc, 0, 2*pi*fs)  # See `calc_G()` docstring.
    # R2   = -R1                        # reflection coefficient looking into ref. impedance
    S21 = G
    # S11  = 2*(R1 + H*R2*G)
    tmp = np.array(list(zip(zip(S11, S21), zip(S21, S11))))
    return rf.Network(s=tmp, f=fs / 1e9, z0=[Zref, Zref])  # `f` is presumed to have units: GHz.


def import_channel(filename, sample_per, fs, zref=100, vic_chnl_ix=1):
    """Read in a channel description file.

    Args:
        filename(str): Name of file from which to import channel description.
        sample_per(real): Sample period of system signal vector (s).
        fs([real]): (Positive only) frequency values being used by caller (Hz).

    KeywordArgs:
        zref(real): Reference impedance (Ohms), for time domain files. (Default = 100)
        vic_chnl_ix(int): Victim channel number (from 1). (Default = 1)

    Returns:
        (skrf.Network, [skrf.Network]): Pair consisting of:
            - a 2-port network description of main channel (i.e. - "victim"), and
            - a list of 2-port network descriptions of aggressors.

    Notes:
        1. When a time domain (i.e. - impulse or step response) file is being imported,
        we have little choice but to use the given reference impedance as the channel
        characteristic impedance, for all frequencies. This implies two things:

            1. Importing time domain descriptions of channels into PyBERT
            yields necessarily lower fidelity results than importing Touchstone descriptions;
            probably not a surprise to those skilled in the art.

            2. The user should take care to ensure that the reference impedance value
            in the GUI is equal to the nominal characteristic impedance of the channel
            being imported when using time domain channel description files.

        2. In the case where a non-empty list of aggressors is returned,
        port 2 of all returned networks corresponds to the same physical node,
        typically, the Rx input node.
    """
    vic_ix = vic_chnl_ix - 1
    assert vic_ix >= 0 and vic_ix < 8, f"Victim index ({vic_ix}) out of range!"
    aggs = []
    extension = os.path.splitext(filename)[1][1:]
    tstone_ext = re.match(r"^s(\d+)p$", extension, re.ASCII | re.IGNORECASE)
    if tstone_ext:  # Touchstone file?
        n_ports = int(tstone_ext.group(1))
        if n_ports == 32:
            ntwks = import_s32p(filename)
            assert ntwks[0].f[-1] < 1e12, f"Maximum frequency > 1 THz!\n\tfs = {fs}\n\tntwks[0] = {ntwks[0]}"
            chnls = list(map(lambda ntwk: interp_s2p(ntwk, fs), ntwks))
            ts2N  = chnls[vic_ix]
            assert ts2N.f[-1] < 1e12, f"Maximum frequency > 1 THz!\n\tfs = {fs}\n\tts2N = {ts2N}"
            ixs = list(range(8))
            ixs.remove(vic_ix)
            aggs = [chnls[ix] for ix in ixs]
        else:
            ts2N = interp_s2p(import_freq(filename), fs)
    else:  # simple 2-column time domain description (impulse or step).
        h = import_time(filename, sample_per)
        # Fixme: an a.c. coupled channel breaks this naive approach!
        if h[-1] > (max(h) / 2.0):  # step response?
            h = diff(h)  # impulse response is derivative of step response.
        Nf = len(fs)
        h.resize(2 * Nf)
        H = fft(h * sample_per)[:Nf]  # Keep the positive frequencies only.
        ts2N = H_2_s2p(H, zref * ones(len(H)), fs, Zref=zref)
    return (ts2N, aggs)


def interp_time(ts, xs, sample_per, n_max=100_000_000):
    """Resample time domain data, using linear interpolation.

    Args:
        ts([float]): Original time values.
        xs([float]): Original signal values.
        sample_per(float): System sample period.

    KeywordArgs:
        n_max(int): Maximum number of elements allowed in resultant vector.
            (Default: 100,000,000)

    Returns:
        [float]: Resampled waveform.
    """
    tmax = ts[-1]
    res = []
    t = 0.0
    i = 0
    while t < tmax and i < n_max:
        while ts[i] <= t:
            i = i + 1
        res.append(xs[i - 1] + (xs[i] - xs[i - 1]) * (t - ts[i - 1]) / (ts[i] - ts[i - 1]))
        t += sample_per
    if len(res) > n_max:
        res = res[:n_max]
    return array(res)


def import_time(filename, sample_per):
    """Read in a time domain waveform file, resampling as appropriate, via
    linear interpolation.

    Args:
        filename(str): Name of waveform file to read in.
        sample_per(float): New sample interval

    Returns:
        [float]: Resampled waveform.
    """
    ts = []
    xs = []
    tmp = []
    with open(filename, mode="rU", encoding="utf-8") as file:
        for line in file:
            try:
                vals = [_f for _f in re.split("[, ;:]+", line) if _f]
                tmp = list(map(float, vals[0:2]))
                ts.append(tmp[0])
                xs.append(tmp[1])
            except:
                # print(f"vals: {vals}; tmp: {tmp}; len(ts): {len(ts)}")
                continue

    return interp_time(ts, xs, sample_per)


# ToDo: Are there SciKit-RF alternatives to these next two functions?
def sdd_21(ntwk, norm=0.5):
    """Given a 4-port single-ended network, return its differential 2-port
    network.

    Args:
        ntwk(skrf.Network): 4-port single ended network.

    KeywordArgs:
        norm(real): Normalization factor. (Default = 0.5)

    Returns:
        skrf.Network: Sdd (2-port).
    """
    mm = se2mm(ntwk)
    return rf.Network(frequency=ntwk.f, s=mm.s[:, 0:2, 0:2], z0=mm.z0[:, 0:2])


def se2mm(ntwk, norm=0.5):
    """Given a 4-port single-ended network, return its mixed mode equivalent.

    Args:
        ntwk(skrf.Network): 4-port single ended network.

    KeywordArgs:
        norm(real): Normalization factor. (Default = 0.5)

    Returns:
        skrf.Network: Mixed mode equivalent network, in the following format:
            Sdd11  Sdd12  Sdc11  Sdc12
            Sdd21  Sdd22  Sdc21  Sdc22
            Scd11  Scd12  Scc11  Scc12
            Scd21  Scd22  Scc21  Scc22
    """
    # Confirm correct network dimmensions.
    (fs, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs == 4, "Touchstone file must have 4 ports!"

    # Detect/correct "1 => 3" port numbering.
    # ix = ntwk.s.shape[0] // 5  # So as not to be fooled by d.c. blocking.
    ix = 1
    if abs(ntwk.s21.s[ix, 0, 0]) < abs(ntwk.s31.s[ix, 0, 0]):  # 1 ==> 3 port numbering?
        ntwk.renumber((1, 2), (2, 1))

    # Convert S-parameter data.
    s = np.zeros(ntwk.s.shape, dtype=complex)
    s[:, 0, 0] = norm * (ntwk.s11 - ntwk.s13 - ntwk.s31 + ntwk.s33).s.flatten()
    s[:, 0, 1] = norm * (ntwk.s12 - ntwk.s14 - ntwk.s32 + ntwk.s34).s.flatten()
    s[:, 0, 2] = norm * (ntwk.s11 + ntwk.s13 - ntwk.s31 - ntwk.s33).s.flatten()
    s[:, 0, 3] = norm * (ntwk.s12 + ntwk.s14 - ntwk.s32 - ntwk.s34).s.flatten()
    s[:, 1, 0] = norm * (ntwk.s21 - ntwk.s23 - ntwk.s41 + ntwk.s43).s.flatten()
    s[:, 1, 1] = norm * (ntwk.s22 - ntwk.s24 - ntwk.s42 + ntwk.s44).s.flatten()
    s[:, 1, 2] = norm * (ntwk.s21 + ntwk.s23 - ntwk.s41 - ntwk.s43).s.flatten()
    s[:, 1, 3] = norm * (ntwk.s22 + ntwk.s24 - ntwk.s42 - ntwk.s44).s.flatten()
    s[:, 2, 0] = norm * (ntwk.s11 - ntwk.s13 + ntwk.s31 - ntwk.s33).s.flatten()
    s[:, 2, 1] = norm * (ntwk.s12 - ntwk.s14 + ntwk.s32 - ntwk.s34).s.flatten()
    s[:, 2, 2] = norm * (ntwk.s11 + ntwk.s13 + ntwk.s31 + ntwk.s33).s.flatten()
    s[:, 2, 3] = norm * (ntwk.s12 + ntwk.s14 + ntwk.s32 + ntwk.s34).s.flatten()
    s[:, 3, 0] = norm * (ntwk.s21 - ntwk.s23 + ntwk.s41 - ntwk.s43).s.flatten()
    s[:, 3, 1] = norm * (ntwk.s22 - ntwk.s24 + ntwk.s42 - ntwk.s44).s.flatten()
    s[:, 3, 2] = norm * (ntwk.s21 + ntwk.s23 + ntwk.s41 + ntwk.s43).s.flatten()
    s[:, 3, 3] = norm * (ntwk.s22 + ntwk.s24 + ntwk.s42 + ntwk.s44).s.flatten()

    # Convert port impedances.
    f = ntwk.f
    z = np.zeros((len(f), 4), dtype=complex)
    z[:, 0] = ntwk.z0[:, 0] + ntwk.z0[:, 2]
    z[:, 1] = ntwk.z0[:, 1] + ntwk.z0[:, 3]
    z[:, 2] = (ntwk.z0[:, 0] + ntwk.z0[:, 2]) / 2
    z[:, 3] = (ntwk.z0[:, 1] + ntwk.z0[:, 3]) / 2

    return rf.Network(frequency=f, s=s, z0=z)


def import_freq(filename):
    """Read in a 1, 2, or 4-port Touchstone file, and return an equivalent
    2-port network.

    Args:
        filename(str): Name of Touchstone file to read in.

    Returns:
        skrf.Network: 2-port network.

    Raises:
        ValueError: If Touchstone file is not 1, 2, or 4-port.

    Notes:
        1. A 4-port Touchstone file is assumed single-ended,
        and the "DD" quadrant of its mixed-mode equivalent gets returned.
    """
    # Import and sanity check the Touchstone file.
    ntwk = rf.Network(filename, f_unit="Hz")
    (fs, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs in (1, 2, 4), f"Touchstone file must have 1, 2, or 4 ports!\n{ntwk}"

    # Convert to a 2-port network.
    if rs == 4:  # 4-port Touchstone files are assumed single-ended!
        return sdd_21(ntwk)
    if rs == 2:
        return ntwk
    return rf.network.one_port_2_two_port(ntwk)


def import_s32p(filename, vic_chnl=1):
    """Read in a 32-port Touchstone file, and return an equivalent list
    of 16 2-port networks: a single victim through channel and
    15 crosstalk aggressors.

    Args:
        filename(str): Name of Touchstone file to read in.

    KeywordArgs:
        vic_chnl(int): Victim channel number (from 1). (Default = 1)

    Returns:
        [skrf.Network]: List of 16 2-port network.
            (First element is the victim.)

    Raises:
        ValueError: If Touchstone file is not 32-port.

    Notes:
        1. Input Touchstone file is assumed single-ended.
        2. The differential through and xtalk channels are returned.
        3. Port 2 of all returned channels correspond to the same
        physical circuit node, typically, the Rx input node.
    """
    # Import and sanity check the Touchstone file.
    ntwk = rf.Network(filename)
    (fs, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs == 32, f"Touchstone file must have 32 ports!\n\t{ntwk}"
    # Check for needed port renumbering.
    if abs(ntwk.s[0,16,0]) > abs(ntwk.s[0,1,0]):  # |S[17,1]| > |S[2,1]|?
        ntwk.renumber( [  0, 16,  1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23
                       ,  8, 24,  9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
                       ]
                     , [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
                       , 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                       ]
                     )
    # Extract the victim and aggressors.
    def ports_from_chnls(left, right):
        """Return list of 4 ports corresponding to a particular left and right
        channel ID (from 1), assuming "1=>2/3=>4" convention.

        Args:
            left(int): Left side channel number (from 1).
            right(int): Right side channel number (from 1).

        Returns:
            [int]: List of ports (from 0) for desired channel.
        """
        return [(left-1)*4, (right-1)*4 + 1, (left-1)*4 + 2, (right-1)*4 + 3]

    vic = sdd_21(rf.subnetwork(ntwk, ports_from_chnls(vic_chnl, vic_chnl)))
    agg_chnls = list(array(range(8)) + 1)
    agg_chnls.remove(vic_chnl)
    aggs = list(map(sdd_21, [rf.subnetwork(ntwk, ports_from_chnls(chnl, vic_chnl)) for chnl in agg_chnls]))
    return [vic] + aggs


def lfsr_bits(taps, seed):
    """Given a set of tap indices and a seed, generate a PRBS.

    Args:
        taps([int]): The set of fed back taps.
                     (Largest determines order of generator.)
        seed(int): The initial value of the shift register.

    Returns:
        generator: A PRBS generator object with a next() method, for retrieving
            the next bit in the sequence.
    """
    val = int(seed)
    num_taps = max(taps)
    mask = (1 << num_taps) - 1

    while True:
        xor_res = reduce(lambda x, b: x ^ b, [bool(val & (1 << (tap - 1))) for tap in taps])
        val = (val << 1) & mask  # Just to keep 'val' from growing without bound.
        if xor_res:
            val += 1
        yield val & 1


def safe_log10(x):
    """Guards against pesky 'Divide by 0' error messages."""

    if hasattr(x, "__len__"):
        x = where(x == 0, 1.0e-20 * ones(len(x)), x)
    else:
        if x == 0:
            x = 1.0e-20

    return log10(x)


def pulse_center(p, nspui):
    """Determines the center of the pulse response, using the "Hula Hoop"
    algorithm (See SiSoft/Tellian's DesignCon 2016 paper.)

    Args:
        p([Float]): The single bit pulse response.
        nspui(Int): The number of vector elements per unit interval.

    Returns:
        (Int, float): The estimated index at which the clock will
            sample the main lobe, and the vertical threshold at which
            the main lobe is UI wide.
    """
    div = 2.0
    p_max = p.max()
    thresh = p_max / div
    main_lobe_ixs = where(p > thresh)[0]
    if not main_lobe_ixs.size:  # Sometimes, the optimizer really whacks out.
        return (-1, 0)  # Flag this, by returning an impossible index.

    err = main_lobe_ixs[-1] - main_lobe_ixs[0] - nspui
    while err and div < 5000:
        div *= 2.0
        if err > 0:
            thresh += p_max / div
        else:
            thresh -= p_max / div
        main_lobe_ixs = where(p > thresh)[0]
        err = main_lobe_ixs[-1] - main_lobe_ixs[0] - nspui

    clock_pos = int(mean([main_lobe_ixs[0], main_lobe_ixs[-1]]))
    return (clock_pos, thresh)


def submodules(package):
    """Find all sub-modules of a package."""
    rst = {}

    for imp, name, _ in pkgutil.iter_modules(package.__path__):
        fullModuleName = f"{package.__name__}.{name}"
        mod = importlib.import_module(fullModuleName, package=package.__path__)
        rst[name] = mod

    return rst


def cap_mag(zs, maxMag=1.0):
    """Cap the magnitude of a list of complex values, leaving the phase
    unchanged.

    Args:
        zs([complex]): The complex values to be capped.

    KeywordArgs:
        maxMag(real): The maximum allowed magnitude. (Default = 1)

    Notes:
        1. Any pre-existing shape of the input will be preserved.
    """
    orig_shape = zs.shape
    zs_flat = zs.flatten()
    subs = [rect(maxMag, phase(z)) for z in zs_flat]
    return where(abs(zs_flat) > maxMag, subs, zs_flat).reshape(zs.shape)


def mon_mag(zs):
    """Enforce monotonically decreasing magnitude in list of complex values,
    leaving the phase unchanged.

    Args:
        zs([complex]): The complex values to be adjusted.

    Notes:
        1. Any pre-existing shape of the input will be preserved.
    """
    orig_shape = zs.shape
    zs_flat = zs.flatten()
    for ix in range(1, len(zs_flat)):
        zs_flat[ix] = rect(min(abs(zs_flat[ix - 1]), abs(zs_flat[ix])), phase(zs_flat[ix]))
    return zs_flat.reshape(zs.shape)


def interp_s2p(ntwk, f):
    """Safely interpolate a 2-port network, by applying certain constraints to
    any necessary extrapolation.

    Args:
        ntwk(skrf.Network): The 2-port network to be interpolated.
        f([real]): The list of new frequency sampling points (Hz).

    Returns:
        skrf.Network: The interpolated/extrapolated 2-port network.

    Raises:
        ValueError: If `ntwk` is _not_ a 2-port network.
    """
    (fs, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs == 2, "Touchstone file must have 2 ports!"

    extrap = ntwk.interpolate(f, fill_value="extrapolate", coords="polar", assume_sorted=True)
    assert extrap.f[-1] < 1e12, f"Maximum frequency > 1 THz!\n\tf: {f}\n\textrap: {extrap}"
    s11 = cap_mag(extrap.s[:, 0, 0])
    s22 = cap_mag(extrap.s[:, 1, 1])
    s12 = ntwk.s12.interpolate(f, fill_value=0, bounds_error=False, coords="polar", assume_sorted=True).s.flatten()
    s21 = ntwk.s21.interpolate(f, fill_value=0, bounds_error=False, coords="polar", assume_sorted=True).s.flatten()
    s = np.array(list(zip(zip(s11, s12), zip(s21, s22))))
    if ntwk.name is None:
        ntwk.name = "s2p"
    return rf.Network(f=f, s=s, z0=extrap.z0, name=(ntwk.name + "_interp"), f_unit="Hz")


# ToDo: Are there any uses of this function remaining? Can we eliminate them?
def renorm_s2p(ntwk, zs):
    """Renormalize a simple 2-port network to a new set of port impedances.

    This function was originally written as a check on the
    `skrf.Network.renormalize()` function, which I was attempting to use
    to model the Rx termination when calculating the channel impulse
    response. (See lines 1640-1650'ish of `pybert.py`.)

    In my original specific case, I was attempting to model an open
    circuit termination. And when I did the magnitude of my resultant
    S21 dropped from 0 to -44 dB!
    I didn't think that could possibly be correct.
    So, I wrote this function as a check on that.

    Args:
        ntwk(skrf.Network): A 2-port network, which must use the same
        (singular) impedance at both ports.

        zs(complex array-like): The set of new port impedances to be
        used. This set of frequencies may be unique for each port and at
        each frequency.

    Returns:
        skrf.Network: The renormalized 2-port network.
    """
    (Nf, Nr, Nc) = ntwk.s.shape
    assert Nr == 2 and Nc == 2, "May only be used to renormalize a 2-port network!"
    assert all(ntwk.z0[:, 0] == ntwk.z0[0, 0]) and all(
        ntwk.z0[:, 0] == ntwk.z0[:, 1]
    ), f"May only be used to renormalize a network with equal (singular) reference impedances! z0: {ntwk.z0}"
    assert zs.shape == (2,) or zs.shape == (
        len(ntwk.f),
        2,
    ), "The list of new impedances must have shape (2,) or (len(ntwk.f), 2)!"

    if zs.shape == (2,):
        zt = zs.repeat(len(Nf))
    else:
        zt = np.array(zs)
    z0 = ntwk.z0[0, 0]
    S = ntwk.s
    I = np.identity(2)
    Z = []
    for s in S:
        Z.append(inv(I - s).dot(I + s))  # Resultant values are normalized to z0.
    Z = np.array(Z)
    Zn = []
    for z, zn in zip(Z, zt):  # Iterration is over frequency and yields: (2x2 array, 2-element vector).
        Zn.append(z.dot(z0 / zn))
    Zn = np.array(Zn)
    Sn = []
    for z in Zn:
        Sn.append(inv(z + I).dot(z - I))
    return rf.Network(s=Sn, f=ntwk.f / 1e9, z0=zs)


def getwave_step_resp(ami_model):
    """Use a model's GetWave() function to extract its step response.

    Args:
        ami_model (): The AMI model to use.

    Returns:
        NumPy 1-D array: The model's step response.

    Raises:
        RuntimeError: When no step rise is detected.
    """
    # Delay the input edge slightly, in order to minimize high
    # frequency artifactual energy sometimes introduced near
    # the signal edges by frequency domain processing in some models.
    tmp = array([-0.5] * 128 + [0.5] * 896)  # Stick w/ 2^n, for freq. domain models' sake.
    tx_s, _ = ami_model.getWave(tmp)
    # Some models delay signal flow through GetWave() arbitrarily.
    tmp = array([0.5] * 1024)
    max_tries = 10
    n_tries = 0
    while max(tx_s) < 0 and n_tries < max_tries:  # Wait for step to rise, but not indefinitely.
        tx_s, _ = ami_model.getWave(tmp)
        n_tries += 1
    if n_tries == max_tries:
        raise RuntimeError("No step rise detected!")
    # Make one more call, just to ensure a sufficient "tail".
    tmp, _ = ami_model.getWave(tmp)
    tx_s = np.append(tx_s, tmp)
    return tx_s - tx_s[0]


def make_bathtub(centers, jit_pdf, min_val=0, rj=0, extrap=False):
    """Generate the "bathtub" curve associated with a particular jitter PDF.

    Args:
        centers([real]): List of uniformly spaced bin centers (s).
        jit_pdf([real]): PDF of jitter.

    KeywordArgs:
        min_val(real): Minimum value allowed in returned bathtub vector.
            Default: 0
        rj(real): Standard deviation of Gaussian PDF characterizing random jitter.
            Default: 0
        extrap(bool): Extrapolate bathtub tails, using `rj`, if True.
            Default: True

    Returns:
        ([real], (int,int)): A pair consisting of:
            - the vector of probabilities forming the bathtub curve, and
            - a pair consisting of the beginning/end indices of the extrapolated region.
    """
    half_len  = len(jit_pdf) // 2
    dt        = centers[1] - centers[0]  # Bins assumed to be uniformly spaced!
    zero_locs = where(fftshift(jit_pdf) == 0)[0]
    ext_first = 0
    ext_last  = len(jit_pdf)
    if (extrap and len(zero_locs)):
        ext_first = min(zero_locs)
        ext_last  = max(zero_locs)
        sqrt_2pi = sqrt(2*pi)
        ix_r = ext_first + half_len - 1
        mu_r = centers[ix_r] - sqrt(2) * rj * sqrt(-log(rj * sqrt_2pi * jit_pdf[ix_r]))
        ix_l = ext_last - half_len + 1
        mu_l = centers[ix_l] + sqrt(2) * rj * sqrt(-log(rj * sqrt_2pi * jit_pdf[ix_l]))
        jit_pdf = append( append( gaus_pdf(centers[:ix_l], mu_l, rj)
                                , jit_pdf[ix_l:ix_r+1])
                        , gaus_pdf(centers[ix_r+1:], mu_r, rj))
    bathtub  = list(cumsum(jit_pdf[-1 : -(half_len+1) : -1]))
    bathtub.reverse()
    bathtub  = array(bathtub + list(cumsum(jit_pdf[: half_len+1]))) * 2*dt
    bathtub  = where(bathtub < min_val, min_val * ones(len(bathtub)), bathtub)
    return (bathtub, (ext_first,ext_last))

def parZ(Z1, Z2):
    """Calculate the parallel combination of two impedances.
    """
    return (Z1 * Z2) / (Z1 + Z2)

def Zcap(w, C):
    """Calculate the impedance of a capacitor, guarding against /0.
    """
    if w[0] == 0:
        return append(array([1e15]), -1j / (w[1:] * C))
    else:
        return -1j / (w * C)

def calc_com(ui, sbr, osf, ser, rlm,
    z, p1, p2, gDC_min, gDC_max, fHP, gHP_min, gHP_max,
    nTx, tx_min, tx_max, nDFE, dfe_min, dfe_max,
    Add, TxSNR, eta0, sigRj,
    Rd, Cd, Cb, Cp, Ls, Zc, td,
    mod=0):
    """Calculate the Channel Operating Margin (COM), given single bit response (SBR).

    Args:
        ui(real): unit interval (s).
        sbr([[real]]): Single bit response, uniformly spaced samples, dimmensions KxN, where:
            K is 1 + the number of aggressors (victim occupies first row), and
            N is the length of the response to be considered here.
        osf(int): Oversampling factor (i.e. - samples per UI).
        ser(real): Target symbol error rate.
        rlm(real): Relative level mismatch (for PAMn).
        z(complex): Zero in CTLE response (Hz).
        p1(complex): First pole in CTLE response (Hz).
        p2(complex): Second pole in CTLE response (Hz).
        gDC_min(real): Minimum d.c. gain of CTLE peaking filter.
        gDC_max(real): Maximum d.c. gain of CTLE peaking filter.
        fHP(real): Frequency of 2nd CTLE stage pole/zero (Hz).
        gHP_min(real): Minimum d.c. gain of CTLE 2nd stage filter.
        gHP_max(real): Maximum d.c. gain of CTLE 2nd stage filter.
        nTx(int): Number of Tx FFE taps.
        tx_min([real]): Minimum Tx FFE weights. (Should have `nTx` elements.)
        tx_max([real]): Maximum Tx FFE weights. (Should have `nTx` elements.)
        nDFE(int): Number of DFE taps.
        dfe_min([real]): Minimum DFE tap weights. (Should have `nDFE` elements.)
        dfe_max([real]): Maximum DFE tap weights. (Should have `nDFE` elements.)
        Add(real): Peak-to-peak dual-Dirac model DJ (UI).
        TxSNR(real): Tx signal-to-noise ratio (dB).
        eta0(real): One sided noise spectral density (V^2/Hz).
        sigRj(real): Random jitter standard deviation (UI) (from dual-Dirac).
        Rd(real): Die source/term. resistance (Ohms).
        Cd(real): Die parasitic capacitance (pF).
        Cb(real): Bump parasitic capacitance (pF).
        Cp(real): Ball parasitic capacitance (pF).
        Ls(real): Series (between Cd and Cb) inductance (nH).
        Zc(real): Characteristic impedance of package TL (Ohms).
        td(real): One way propagation delay of package TL (ps).

    KeywordArgs:
        mod(int): Modulation type (0=NRZ, 1=PAM4).
            Default: 0 (NRZ)

    Returns:
        (real, real, [real], [real], int, [real], [real], real, [real], {}): A tuple consisting of:
            - Asig: Main cursor signal amplitude.
            - Anoise_xtalk: Eye closure, due to noise & crosstalk.
            - pmf: Probability mass function of noise & crosstalk.
            - cmf: Cumulative mass function of noise & crosstalk.
            - loc: Index at which `cmf` rises above desired SER.
            - sbr_eq: Equalized single bit response.
            - tx_taps: Optimized Tx FFE tap weights.
            - ctle_G: Optimized d.c. gain of CTLE respone.
            - dfe_taps: Optimized Rx DFE tap weights.
            - opt_rslts: Dictionary containing various post-optimization results.
    """
    # Perform any unit conversion required both here and in `com_opt()`.
    Add  *= ui
    varRj = (sigRj*ui)**2
    # Determine set of possible signal levels, based on modulation scheme.
    if mod:
        levels = [-1, -0.3333, 0.3333, 1]
    else:
        levels = [-1, 1]
    # Construct CTLE generator.
    ctle_gen = make_ctle_gen(z, p1, p2, fHP)
    # Construct apropos time/frequency vectors.
    ts = ui / osf            # sampling interval
    n_samps = len(sbr[0])    # number of samples in victim SBR
    f0 = 1 / (ts * n_samps)  # fundamental frequency
    # "+1" because `irfft()` expects the Nyquist component tacked onto the end of its input vector.
    f  = array([n*f0 for n in range(n_samps//2 + 1)])
    w  = 2*pi*f
    s  = 1j*w
    # Construct Rx input filter.
    r   = s / (2*pi * 0.75/ui)                            # As per COM, f_cutoff = 0.75 f_symb.
    Hrx = 105 / (r**4 + 10*r**3 + 45*r**2 + 105*r + 105)  # 4th-order Bessel-Thomson LP
    # Calculate package response.
    Zchnl = 100  # ToDo: Do we have this after a PyBERT run?
    # Parallel combination of ball parasitic capacitance and channel input impedance.
    Zload = parZ(Zcap(w, Cp*1e-12/2), Zchnl)    
    Rout  = (Zload - Zc)/(Zload + Zc)           # Reflection coefficient exiting the TL.
    Rout_dly = Rout*exp(-2j*w*td*1e-12)         # Above plus one round-trip phase delay.
    Zin   = Zc * (1 + Rout_dly)/(1 - Rout_dly)  # Impedance looking into the TL.
    Ztot  = 2*Rd + parZ(Zcap(w, Cd*1e-12/2), 1j*w*2*Ls*1e-9 + parZ(Zcap(w, Cb*1e-12/2), Zin))
    Hpkg  = Zload / Ztot
    # Find optimal equalization scheme.
    tx_taps, ctle_gain, hp_gain, dfe_taps, opt_rslts = com_opt(
        w, sbr, ui, osf, rlm, Hrx * Hpkg**2,
        ctle_gen, gDC_min, gDC_max, gHP_min, gHP_max,
        nTx, tx_min, tx_max, nDFE, dfe_min, dfe_max,
        Add, TxSNR, varRj, eta0, mod=mod, max_iter=5000)
    sbr_opt   = opt_rslts['sbr_eq']
    cur_ix    = opt_rslts['cur_ix']
    Asig      = opt_rslts['Asig']
    varTx     = opt_rslts['varTx']
    varISI    = opt_rslts['varISI']
    mu_n      = opt_rslts['mu_n']
    varJ      = opt_rslts['varJ']
    varSpec   = opt_rslts['varSpec']
    dISI_n    = opt_rslts['dISI']
    # Stash some intermediate results that get plotted.
    opt_rslts['Hrx']  = Hrx
    opt_rslts['Hpkg'] = Hpkg
    opt_rslts['f']    = f
    # Make final calculations of distributions and noises.
    dj_noises = Add * mu_n
    opt_rslts['dj_noises'] = dj_noises
    if len(sbr) > 1:
        deltas = [ level*Asig + dj_noise + dISI + xtalk
                     for level    in levels
                     for dj_noise in dj_noises
                     for dISI     in dISI_n
                     for xtalk    in concatenate([agg[cur_ix::osf] for agg in sbr[1:]])
                 ]
    else:       
        deltas = [ level*Asig + dj_noise + dISI
                     for level    in levels
                     for dj_noise in dj_noises
                     for dISI     in dISI_n
                 ]
    varG = varTx + varSpec + sum(varRj * mu_n)
    opt_rslts['deltas'] = deltas
    opt_rslts['varG']   = varG
    gaus = array([exp(-(y**2 / varG)) for y in linspace(-0.5, 0.5, num=1001)])
    pmf = array(sum([np.roll(gaus, int(delta*1000)) for delta in deltas]))
    pmf = pmf / sum(pmf)
    cmf = cumsum(pmf)
    try:
        if mod:  # PAM4
            _cmf = abs(cmf*4 - 2)
        else:    # NRZ
            _cmf = abs(cmf*2 - 1)
        open_ixs = where(_cmf < ser)[0]
        left_ix  = open_ixs[0]
        right_ix = open_ixs[-1]
        if (right_ix - 500) < (500 - left_ix):
            ix = right_ix
        else:
            ix = left_ix
    except:
        print(pmf)
        ix = 190
    Anoise_xtalk = Asig - abs(ix - 500) * 0.001
    if Anoise_xtalk == 0:
        print(f"Asig: {Asig}, varG: {varG}, min(deltas): {min(deltas)}, max(deltas): {max(deltas)}\ndISI_n: {dISI_n}")

    return (Asig, Anoise_xtalk, pmf, cmf, ix, sbr_opt, tx_taps, ctle_gain, hp_gain, dfe_taps, opt_rslts)

def com_opt(w, sbr, ui, osf, rlm, Hdev, ctle_gen, gDC_min, gDC_max, gHP_min, gHP_max,
    nTx, tx_min, tx_max, nDFE, dfe_min, dfe_max,
    Add, TxSNR, varRj, eta0, mod=0, max_iter=50):
    """Optimize EQ for COM calculation.
    
    Args:
        w([real]): Angular frequencies of interest.
        sbr([[real]]): Single bit response, uniformly spaced samples, dimmensions KxN, where:
            K is 1 + the number of aggressors (victim occupies first row), and
            N is the length of the response to be considered here.
        ui(real): unit interval (s).
        osf(int): Oversampling factor (i.e. - samples per UI).
        rlm(real): Relative level mismatch (for PAM4).
        Hdev([complex]): Combined frequency response of Rx input and two packages.
        ctle_gen(function): CTLE generator (See `make_ctle_gen()`.).
        gDC_min(real): Minimum d.c. gain of CTLE peaking filter.
        gDC_max(real): Maximum d.c. gain of CTLE peaking filter.
        gHP_min(real): Minimum d.c. gain of CTLE 2nd stage filter.
        gHP_max(real): Maximum d.c. gain of CTLE 2nd stage filter.
        nTx(int): Number of Tx FFE taps.
        tx_min([real]): Minimum Tx FFE weights. (Should have `nTx` elements.)
        tx_max([real]): Maximum Tx FFE weights. (Should have `nTx` elements.)
        nDFE(int): Number of DFE taps.
        dfe_min([real]): Minimum DFE tap weights. (Should have `nDFE` elements.)
        dfe_max([real]): Maximum DFE tap weights. (Should have `nDFE` elements.)
        Add(real): Peak-to-peak dual-Dirac model DJ (s).
        TxSNR(real): Tx signal-to-noise ratio (dB).
        varRj(real): Random jitter variance (from dual-Dirac).
        eta0(real): One sided noise spectral density (V^2/Hz).

    KeywordArgs:
        mod(int): Modulation type (0=NRZ, 1=PAM4).
            Default: 0 (NRZ)
        max_iter(int): Maximum optimization iterations.
            Default: 50

    Returns:
        ([real], real, [real], {}): Tuple consisting of:
            - optimum Tx FFE tap weights,
            - optimum CTLE gain,
            - optimum DFE tap weights, and
            - a dictionary of certain internal optimization results.
    """
    n_samps = len(sbr[0])
    opt_rslts = {}
    def fom(args):
        """Objective function to be optimized. (See COM spec.)
        """
        tx_taps   = args[:nTx]
        ctle_gain = args[nTx]
        hp_gain   = args[nTx+1]
        dfe_taps  = args[nTx+2:]
        tx_ffe_h  = np.zeros(osf*nTx, dtype=tx_taps.dtype)
        tx_ffe_h[::osf] = tx_taps
        tx_ffe_H  = rfft(tx_ffe_h, n=n_samps)
        tx_ffe_H *= sum(tx_taps) / abs(tx_ffe_H[0])
        ctle_H    = ctle_gen(ctle_gain, hp_gain, w)
        ctle_h    = irfft(ctle_H)
        ctle_h    = roll(ctle_h, (nTx-2)*osf)       # Line up w/ Tx FFE main tap.
        ctle_s    = cumsum(ctle_h)                  # Use step response to scale impulse response,
        ctle_h   *= abs(ctle_H[0]) / ctle_s[-1]     # since we know our desired d.c. gain.
        Heq       = Hdev * tx_ffe_H * ctle_H        # Complete freq. resp. of Tx/Rx pair.
        heq       = irfft(Heq)
        seq       = cumsum(heq)                     # Use step response to scale impulse response,
        heq      *= abs(Heq[0]) / seq[-1]           # since we know our desired d.c. gain.
        sbr_eq    = convolve(sbr[0], heq)[:n_samps]
        cur_ix, ui_height = pulse_center(sbr_eq, osf)
        Asig      = sbr_eq[cur_ix]  # ToDo: Do we need to scale this?
        sbr_tail  = sbr_eq[cur_ix+osf::osf]
        dISI      = sbr_tail - pad(dfe_taps*Asig, (0, len(sbr_tail) - len(dfe_taps)),
                                     mode='constant', constant_values=0)
        if mod:
            Asig *= 1/3 * rlm
        varTx  = (Asig / pow(10, TxSNR/20))**2
        varISI = sum(dISI**2)
        if mod:
            varISI *= 5/9
        mu_n = diff(sbr_eq)[cur_ix+osf-1::osf]
        varJ = sum((Add**2 + varRj) * mu_n**2)
        if mod:
            varJ *= 5/9
        varSpec  = sum(abs(ctle_H)**2) * eta0
        varXtalk = sum([sum(agg[cur_ix::osf]**2) for agg in sbr[1:]])
        if mod:
            varXtalk *= 5/9

        opt_rslts['sbr_eq']  = sbr_eq
        opt_rslts['cur_ix']  = cur_ix
        opt_rslts['Asig']    = Asig
        opt_rslts['varTx']   = varTx
        opt_rslts['varISI']  = varISI
        opt_rslts['mu_n']    = mu_n
        opt_rslts['varJ']    = varJ
        opt_rslts['varSpec'] = varSpec
        opt_rslts['dISI']    = dISI
        opt_rslts['tx_ffe_h'] = tx_ffe_h
        opt_rslts['tx_ffe_H'] = tx_ffe_H
        opt_rslts['ctle_h']   = ctle_h
        opt_rslts['ctle_H']   = ctle_H
        opt_rslts['Htot']     = Heq

        # Returned result is negated, because `scipy.optimize` only offers a `minimize()` function.
        return -Asig**2 / (varTx + varISI + varJ + varXtalk + varSpec)

    bounds = list(zip(tx_min, tx_max)) \
           + [(pow(10, gDC_min/20), pow(10, gDC_max/20))] + [(pow(10, gHP_min/20), pow(10, gHP_max/20))] \
           + list(zip(dfe_min, dfe_max))
    cons   = {"type": "ineq", "fun": lambda x: 1 - sum(abs(x[:nTx]))}
    res = minimize(fom, append(append(append(zeros(nTx - 2), array([1, 0])), array([1,1])), zeros(nDFE)),
        bounds=bounds,
        constraints=cons,
        method="trust-constr",
        options={"disp": False, "maxiter": max_iter},
    )
    if not res.success:
        print(f"Optimizer failed with: {res.message}")
    tx_taps   = res.x[:nTx]
    ctle_gain = res.x[nTx]
    hp_gain   = res.x[nTx+1]
    dfe_taps  = res.x[nTx+2:]

    return (tx_taps, ctle_gain, hp_gain, dfe_taps, opt_rslts)


# ToDo: Finish this and use it throughout.
#
# t_and_f_from_ntwk(..., fstep=None) instead? (Would accommodate non-uniform frequency vector.)
def t_from_f(f):
    """Calculate time vector
    """
    raise RuntimeError("Not yet implemented.")


def calc_sbr(s2p, ui, Zt=None):
    """Calculate the single bit response of a 2-port network.

    Args:
        s2p(skrf.Network): 2-port network to convert.
        ui(real): Unit interval (s).

    KeywordArgs:
        Zt(complex or [complex]): Termination impedance, either scalar or as function of frequency.
            (Default: None)

    Returns:
        ([real], [real]): Pair consisting of:
            t: Time vector associated w/ calculated SBR (s).
            sbr: Single bit response (SBR) (a.k.a. - pulse response) of network.
    """
    ts = 1/(2 * s2p.f[-1])  # Sampling frequency = twice Nyquist.
    t = array([n*ts for n in range(2 * (len(s2p.f) - 1))])  # `f` carries an extra element, for Nyquist.
    dt = t[1] - t[0]
    dn = round(ui/dt)
    h, H = calc_s2p_resp(s2p, Zt)
    s = cumsum(h)
    assert dn < len(s), f"dn ({dn}) is too large!"

    s_shift = pad(s[:-dn], (dn, 0), mode="constant", constant_values=0)
    return (t, s - s_shift)


def calc_s2p_resp(s2p, Zt=None):
    """Calculate the impulse and frequency response of an arbitrarily terminated 2-port network.

    Args:
        s2p(skrf.Network): 2-port network representing the channel.

    KeywordArgs:
        Zt(complex or [complex]): Termination impedance, either scalar or as function of frequency.

    Returns:
        ([real], [complex]): A pair consisting of:
            h: impulse response, and
            H: frequency response.
    """
    s2p_term = s2p.copy()
    if Zt is not None:
        assert len(Zt) == 1 or len(Zt) == len(s2p.f), \
            "Zt must be a scalar or have the same length as network frequency vector!"
        s2p_term_z0 = s2p.z0.copy()
        s2p_term_z0[:, 1] = Zt
        s2p_term.renormalize(s2p_term_z0)
    # We take the transfer function, H, to be a ratio of voltages.
    # So, we must normalize our (now generalized) S-parameters.
    H = s2p_term.s21.s.flatten() * np.sqrt(s2p_term.z0[:, 1] / s2p_term.z0[:, 0])
    h = irfft(H)
    s = cumsum(h)
    h *= abs(H[0]) / s[-1]  # Scale for proper d.c. value.
    return (h, H)
