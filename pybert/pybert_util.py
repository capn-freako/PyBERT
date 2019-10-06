"""
General purpose utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 27, 2014 (Copied from pybert_cntrl.py.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
import os.path
import re
from functools import reduce
import pkgutil
import importlib

import numpy as np
from numpy import (
    array,
    concatenate,
    convolve,
    cumsum,
    diff,
    float,
    histogram,
    insert,
    log10,
    mean,
    ones,
    pi,
    power,
    real,
    reshape,
    resize,
    sign,
    sort,
    sqrt,
    where,
    zeros,
)
from numpy.fft import fft, ifft
from scipy.signal import freqs, get_window, invres
from scipy.stats import norm
import skrf as rf

debug = False
gDebugOptimize = False
gMaxCTLEPeak = 20  # max. allowed CTLE peaking (dB) (when optimizing, only)


def moving_average(a, n=3):
    """
    Calculates a sliding average over the input vector.

    Args:
        a([float]): Input vector to be averaged.
        n(int): Width of averaging window, in vector samples. (Optional;
            default = 3.)

    Returns: the moving average of the input vector, leaving the input
        vector unchanged.
    """

    ret = cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return insert(ret[n - 1 :], 0, ret[n - 1] * ones(n - 1)) / n


def find_crossing_times(
    t, x, min_delay: float = 0.0, rising_first: bool = True, min_init_dev: float = 0.1, thresh: float = 0.0
):
    """
    Finds the threshold crossing times of the input signal.

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

    Returns: an array of signal threshold crossing times.
    """

    if len(t) != len(x):
        raise ValueError("len(t) (%d) and len(x) (%d) need to be the same." % (len(t), len(x)))

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
        assert min_delay < xings[-1], "min_delay ({}) must be less than last crossing time ({}).".format(
            min_delay, xings[-1]
        )
        while xings[i] < min_delay:
            i += 1

    if debug:
        print("min_delay: {}".format(min_delay))
        print("rising_first: {}".format(rising_first))
        print("i: {}".format(i))
        print("max_mag_x: {}".format(max_mag_x))
        print("min_mag_x: {}".format(min_mag_x))
        print("xings[0]: {}".format(xings[0]))
        print("xings[i]: {}".format(xings[i]))

    try:
        if rising_first and diff_sign_x[xing_ix[i]] < 0.0:
            i += 1
    except:
        print("len(diff_sign_x):", len(diff_sign_x))
        print("len(xing_ix):", len(xing_ix))
        print("i:", i)
        raise

    return array(xings[i:])


def find_crossings(t, x, amplitude, min_delay: float = 0.0, rising_first: bool = True, min_init_dev=0.1, mod_type=0):
    """
    Finds the crossing times in a signal, according to the modulation type.

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
            stream with [0, 1, 1], in order to provide a known
            synchronization point for jitter analysis.
            (Optional; default = True.)
        min_init_dev(float): The minimum initial deviation from zero,
            which must be detected, before searching for crossings.
            Normalized to maximum input signal magnitude.
            (Optional; default = 0.1.)
        mod_type(int): The modulation type. Allowed values are:
            {0: NRZ, 1: Duo-binary, 2: PAM-4}
            (Optional; default = 0.)

    Returns: The signal threshold crossing times.
    """

    assert mod_type >= 0 and mod_type <= 2, "ERROR: pybert_util.find_crossings(): Unknown modulation type: {}".format(
        mod_type
    )

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


def calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh=6, num_bins=99, zero_mean=True):
    """
    Calculate the jitter in a set of actual zero crossings, given the ideal crossings and unit interval.

    Inputs:

      - ui               : The nominal unit interval.
      - nui              : The number of unit intervals spanned by the input signal.
      - pattern_len      : The number of unit intervals, before input symbol stream repeats.
      - ideal_xings      : The ideal zero crossing locations of the edges.
      - actual_xings     : The actual zero crossing locations of the edges.
      - rel_thresh       : (optional) The threshold for determining periodic jitter spectral components (sigma).
      - num_bins         : (optional) The number of bins to use, when forming histograms.
      - zero_mean        : (optional) Force the mean jitter to zero, when True.

    Outputs:

      - jitter   : The total jitter.
      - t_jitter : The times (taken from 'ideal_xings') corresponding to the returned jitter values.
      - isi      : The peak to peak jitter due to intersymbol interference.
      - dcd      : The peak to peak jitter due to duty cycle distortion.
      - pj       : The peak to peak jitter due to uncorrelated periodic sources.
      - rj       : The standard deviation of the jitter due to uncorrelated unbounded random sources.
      - tie_ind  : The data independent jitter.
      - thresh   : Threshold for determining periodic components.
      - jitter_spectrum  : The spectral magnitude of the total jitter.
      - tie_ind_spectrum : The spectral magnitude of the data independent jitter.
      - spectrum_freqs   : The frequencies corresponding to the spectrum components.
      - hist        : The histogram of the actual jitter.
      - hist_synth  : The histogram of the extrapolated jitter.
      - bin_centers : The bin center values for both histograms.

    """

    def my_hist(x):
        """
        Calculates the probability mass function (PMF) of the input vector,
        enforcing an output range of [-UI/2, +UI/2], sweeping everything in [-UI, -UI/2] into the first bin,
        and everything in [UI/2, UI] into the last bin.
        """
        hist, bin_edges = histogram(
            x, [-ui] + [-ui / 2.0 + i * ui / (num_bins - 2) for i in range(num_bins - 1)] + [ui]
        )
        bin_centers = (
            [-ui / 2.0] + [mean([bin_edges[i + 1], bin_edges[i + 2]]) for i in range(len(bin_edges) - 3)] + [ui / 2.0]
        )

        return (array(list(map(float, hist))) / sum(hist), bin_centers)

    # Check inputs.
    if not ideal_xings.all():
        raise ValueError("calc_jitter(): zero length ideal crossings vector received!")
    if not actual_xings.all():
        raise ValueError("calc_jitter(): zero length actual crossings vector received!")

    # Line up first ideal/actual crossings, and count/validate crossings per pattern.
    ideal_xings = array(ideal_xings) - (ideal_xings[0] - ui / 2.0)
    actual_xings = array(actual_xings) - (actual_xings[0] - ui / 2.0)
    xings_per_pattern = where(ideal_xings > (pattern_len * ui))[0][0]
    if xings_per_pattern % 2 or not xings_per_pattern:
        print("xings_per_pattern:", xings_per_pattern)
        print("len(ideal_xings):", len(ideal_xings))
        print("min(ideal_xings):", min(ideal_xings))
        print("max(ideal_xings):", max(ideal_xings))
        raise AssertionError("pybert_util.calc_jitter(): Odd number of (or, no) crossings per pattern detected!")
    num_patterns = nui // pattern_len

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
            jitter.append(3.0 * ui / 4.0)  # Pad the jitter w/ alternating +/- 3UI/4.
            jitter.append(-3.0 * ui / 4.0)  # (Will get pulled into [-UI/2, UI/2], later.
            skip_next_ideal_xing = True  # If we missed one, we missed two.
        else:  # Noise may produce several crossings. We find all those
            xings = []  # within the interval [-UI/2, +UI/2] centered
            j = i  # around the ideal crossing, and take the average.
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
    tie_risings = jitter.take(list(range(0, len(jitter), 2)))
    tie_fallings = jitter.take(list(range(1, len(jitter), 2)))
    tie_risings.resize(num_patterns * xings_per_pattern // 2)
    tie_fallings.resize(num_patterns * xings_per_pattern // 2)
    tie_risings = reshape(tie_risings, (num_patterns, xings_per_pattern // 2))
    tie_fallings = reshape(tie_fallings, (num_patterns, xings_per_pattern // 2))

    # - Use averaging to remove the uncorrelated components, before calculating data dependent components.
    try:
        tie_risings_ave = tie_risings.mean(axis=0)
        tie_fallings_ave = tie_fallings.mean(axis=0)
        isi = max(tie_risings_ave.ptp(), tie_fallings_ave.ptp())
    except:
        print("xings_per_pattern:", xings_per_pattern)
        print("len(ideal_xings):", len(ideal_xings))
        raise
    isi = min(isi, ui)  # Cap the ISI at the unit interval.
    dcd = abs(mean(tie_risings_ave) - mean(tie_fallings_ave))

    # - Subtract the data dependent jitter from the original TIE track, in order to yield the data independent jitter.
    tie_ave = sum(list(zip(tie_risings_ave, tie_fallings_ave)), ())
    tie_ave = resize(tie_ave, len(jitter))
    tie_ind = jitter - tie_ave

    # - Use spectral analysis to help isolate the periodic components of the data independent jitter.
    # -- Calculate the total jitter spectrum, for display purposes only.
    # --- Make vector uniformly sampled in time, via zero padding where necessary.
    # --- (It's necessary to keep track of those elements in the resultant vector, which aren't paddings; hence, 'valid_ix'.)
    x, valid_ix = make_uniform(t_jitter, jitter, ui, nui)
    y = fft(x)
    jitter_spectrum = abs(y[: len(y) // 2]) / sqrt(len(jitter))  # Normalized, in order to make power correct.
    f0 = 1.0 / (ui * nui)
    spectrum_freqs = [i * f0 for i in range(len(y) // 2)]

    # -- Use the data independent jitter spectrum for our calculations.
    tie_ind_uniform, valid_ix = make_uniform(t_jitter, tie_ind, ui, nui)

    # --- Normalized, in order to make power correct, since we grab Rj from the freq. domain.
    # --- (I'm using the length of the vector before zero padding, because zero padding doesn't add energy.)
    # --- (This has the effect of making our final Rj estimate more conservative.)
    y = fft(tie_ind_uniform) / sqrt(len(tie_ind))
    y_mag = abs(y)
    y_mean = moving_average(y_mag, n=len(y_mag) // 10)
    y_var = moving_average((y_mag - y_mean) ** 2, n=len(y_mag) // 10)
    y_sigma = sqrt(y_var)
    thresh = y_mean + rel_thresh * y_sigma
    y_per = where(y_mag > thresh, y, zeros(len(y)))  # Periodic components are those lying above the threshold.
    y_rnd = where(y_mag > thresh, zeros(len(y)), y)  # Random components are those lying below.
    y_rnd = abs(y_rnd)
    rj = sqrt(mean((y_rnd - mean(y_rnd)) ** 2))
    tie_per = real(ifft(y_per)).take(valid_ix) * sqrt(len(tie_ind))  # Restoring shape of vector to its original,
    pj = tie_per.ptp()  # non-uniformly sampled state.

    # --- Save the spectrum, for display purposes.
    tie_ind_spectrum = y_mag[: len(y_mag) // 2]

    # - Reassemble the jitter, excluding the Rj.
    # -- Here, we see why it was necessary to keep track of the non-padded elements with 'valid_ix':
    # -- It was so that we could add the average and periodic components back together,
    # -- maintaining correct alignment between them.
    if len(tie_per) > len(tie_ave):
        tie_per = tie_per[: len(tie_ave)]
    if len(tie_per) < len(tie_ave):
        tie_ave = tie_ave[: len(tie_per)]
    jitter_synth = tie_ave + tie_per

    # - Calculate the histogram of original, for comparison.
    hist, bin_centers = my_hist(jitter)

    # - Calculate the histogram of everything, except Rj.
    hist_synth, bin_centers = my_hist(jitter_synth)

    # - Extrapolate the tails by convolving w/ complete Gaussian.
    rv = norm(loc=0.0, scale=rj)
    rj_pdf = rv.pdf(bin_centers)
    rj_pmf = rj_pdf / sum(rj_pdf)
    hist_synth = convolve(hist_synth, rj_pmf)
    tail_len = (len(bin_centers) - 1) // 2
    hist_synth = (
        [sum(hist_synth[: tail_len + 1])]
        + list(hist_synth[tail_len + 1 : len(hist_synth) - tail_len - 1])
        + [sum(hist_synth[len(hist_synth) - tail_len - 1 :])]
    )

    return (
        jitter,
        t_jitter,
        isi,
        dcd,
        pj,
        rj,
        tie_ind,
        thresh[: len(thresh) // 2],
        jitter_spectrum,
        tie_ind_spectrum,
        spectrum_freqs,
        hist,
        hist_synth,
        bin_centers,
    )


def make_uniform(t, jitter, ui, nbits):
    """
    Make the jitter vector uniformly sampled in time, by zero-filling where necessary.

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
    """
    Calculates propagation constant from cross-sectional parameters.

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

    return (gamma, Zc)


def calc_gamma_RLGC(R, L, G, C, ws):
    """
    Calculates propagation constant from R, L, G, and C.

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
    Zc    = sqrt((1j * w * L0 + R) / (1j * w * C + G))  # characteristic impedance (Ohms)

    return (gamma, Zc)


def calc_G(H, Rs, Cs, Zc, RL, Cp, CL, ws):
    """
    Calculates fully loaded transfer function of complete channel.

    Inputs:
      - H     unloaded transfer function of interconnect
      - Rs    source series resistance
      - Cs    source parallel (parasitic) capacitance
      - Zc    frequency dependent characteristic impedance of the interconnect
      - RL    load resistance (differential)
      - Cp    load parallel (parasitic) capacitance (single ended)
      - CL    load series (d.c. blocking) capacitance (single ended)
      - ws    frequency sample points vector

    Outputs:
      - G     frequency dependent transfer function of channel
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    # Impedance looking back into the Tx output is a simple parallel RC network.
    Zs = Rs / (1.0 + 1j * w * Rs * Cs)
    # Rx load impedance is 2 series, a.c.-coupling capacitors, in series w/ parallel comb. of Rterm & parasitic cap.
    # (The two parasitic capacitances are in series.)
    ZL = 2.0 * 1.0 / (1j * w * CL) + RL / (1.0 + 1j * w * RL * Cp / 2)
    # Admittance into the interconnect is (Cs || Zc) / (Rs + (Cs || Zc)).
    Cs_par_Zc = Zc / (1.0 + 1j * w * Zc * Cs)
    A = Cs_par_Zc / (Rs + Cs_par_Zc)
    # Reflection coefficient at Rx:
    R1 = (ZL - Zc) / (ZL + Zc)
    # Reflection coefficient at Tx:
    R2 = (Zs - Zc) / (Zs + Zc)
    # Fully loaded channel transfer function:
    G = A * H * (1 + R1) / (1 - R1 * R2 * H ** 2)
    G = G * (((RL / (1j * w * Cp / 2)) / (RL + 1 / (1j * w * Cp / 2))) / ZL)  # Corrected for divider action.
    # (i.e. - We're interested in what appears across RL.)
    return G


def calc_eye(ui, samps_per_ui, height, ys, y_max, clock_times=None):
    """
    Calculates the "eye" diagram of the input signal vector.

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

    Returns: The "heat map" representing the eye diagram. Each grid
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
            for (samp1, samp2) in zip(
                ys[start_ix : start_ix + 2 * samps_per_ui], ys[start_ix + 1 : start_ix + 1 + 2 * samps_per_ui]
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

def draw_channel(height, width, thickness, separation, ch_type, arraySize=[100, 100]):
    """
    Draws the channel cross section.

    Args:
        height(float): dielectric thickness
        width(float): trace width
        thickness(float): trace thickness
        ch_type(Enum): channel cross-section type

    Keyword Args:
        arraySize([int,int]): Size of image array. (100,100)

    Returns: A drawing of the channel cross-section.
    """

    # Calculate drawing dimmensions.
    [yMax, xMax] = arraySize
    hScale = xMax / 2.0  # Fixed physical bounds, for now.
    vScale = yMax / 0.5
    xMid = int(xMax // 2)
    yMid = int(yMax // 2)
    yOff = int(yMax // 10 + vScale * height)
    if ch_type == 'microstrip_se' or ch_type == 'stripline_se':  # single-ended configuration
        xOff1 = xMid - int(hScale * (width/2))
        xOff2 = None
    else :                                                       # differential configuration
        xOff1 = xMid - int(hScale * (separation/2 + width))
        xOff2 = xMid + int(hScale * separation/2)
    # Generate the cross-section drawing.
    # - Fill w/ dielectric color.
    img_array = 10 * ones(arraySize)
    # - Draw the reference plane.
    for y in range(yMax // 10):
        for x in range(xMax):
            img_array[y,x] = 0
    # - Draw air, or second plane, depending on configuration.
    if ch_type == 'microstrip_se' or ch_type == 'microstrip_diff':  # microstrip configuration
        for y in range(yOff, yMax):
            for x in range(xMax):
                img_array[y, x] = 255  # air (white)
    else :                                                          # stripline configuration
        for y in range(yOff + int(vScale*(thickness + height)), yMax):
            for x in range(xMax):
                img_array[y, x] = 0  # metal (black)
    # - Draw trace(s).
    for y in range(int(vScale * thickness)):
        for x in range(int(hScale * width)):
            img_array[yOff + y, xOff1 + x] = 0
            if xOff2:
                img_array[yOff + y, xOff2 + x] = 0
    return img_array

def make_ctle(rx_bw, peak_freq, peak_mag, w, mode="Passive", dc_offset=0):
    """
    Generate the frequency response of a continuous time linear
    equalizer (CTLE), given the:

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
        raise RuntimeError("pybert_util.make_ctle(): Unrecognized value for 'mode' parameter: {}.".format(mode))

    return (w, H)


def trim_impulse(g, min_len=0, max_len=1000000):
    """
    Trim impulse response, for more useful display, by:
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
    Pt = 0.998 * sum(g ** 2)
    i = 0
    P = 0
    while P < Pt:
        P += g[i] ** 2
        i += 1
    stop_ix = min(max_ix + max_len, max(i, max_ix + min_len))

    # Set "front porch" to 20%, guarding against negative start index.
    start_ix = max(0, max_ix - (stop_ix - max_ix) // 4)

    return (g[start_ix:stop_ix], start_ix)


def import_channel(filename, sample_per, padded=False, windowed=False):
    """
    Read in a channel file.

    Args:
        filename(str): Name of file from which to import channel description.
        sample_per(float): Sample period of signal vector (s).
        padded(Bool): (Optional) Zero pad s4p data, such that fmax >= 1/(2*sample_per)? (Default = False)
        windowed(Bool): (Optional) Window s4p data, before converting to time domain? (Default = False)

    Returns: Imported channel impulse, or step, response.
    """

    extension = os.path.splitext(filename)[1][1:]
    if extension in ("s4p", "S4P"):
        return import_freq(filename, sample_per, padded=padded, windowed=windowed)
    return import_time(filename, sample_per)


def interp_time(ts, xs, sample_per):
    """
    Resample time domain data, using linear interpolation.

    Args:
        ts([float]): Original time values.
        xs([float]): Original signal values.
        sample_per(float): System sample period.

    Returns: Resampled waveform.
    """
    tmax = ts[-1]
    res = []
    t = 0.0
    i = 0
    while t < tmax:
        while ts[i] <= t:
            i = i + 1
        res.append(xs[i - 1] + (xs[i] - xs[i - 1]) * (t - ts[i - 1]) / (ts[i] - ts[i - 1]))
        t += sample_per

    return array(res)


def import_time(filename, sample_per):
    """
    Read in a time domain waveform file, resampling as
    appropriate, via linear interpolation.

    Args:
        filename(str): Name of waveform file to read in.
        sample_per(float): New sample interval

    Returns: Resampled waveform.
    """

    # Read in original data from file.
    ts = []
    xs = []
    with open(filename, mode="rU") as file:
        for line in file:
            try:
                tmp = list(map(float, [_f for _f in re.split("[, ;:]+", line) if _f][0:2]))
            except:
                continue
            ts.append(tmp[0])
            xs.append(tmp[1])

    return interp_time(ts, xs, sample_per)


def sdd_21(ntwk):
    """
    Given a 4-port single-ended network, return its differential throughput.

    Args:
        ntwk(skrf.Network): 4-port single ended network.

    Returns: Sdd[2,1].
    """

    if real(ntwk.s21.s[0, 0, 0]) < 0.5:  # 1 ==> 3 port numbering?
        ntwk.renumber((2, 3), (3, 2))

    return 0.5 * (ntwk.s21 - ntwk.s23 + ntwk.s43 - ntwk.s41)


def import_freq(filename, sample_per, padded=False, windowed=False, f_step=10e6):
    """
    Read in a single ended 4-port Touchstone file, and extract the
    differential throughput step response, resampling as
    appropriate, via linear interpolation.

    Args:
        filename(str): Name of Touchstone file to read in.
        sample_per(float): New sample interval
        padded(Bool): (Optional) Zero pad s4p data, such that fmax >= 1/(2*sample_per)? (Default = False)
        windowed(Bool): (Optional) Window s4p data, before converting to time domain? (Default = False)

    Returns: Resampled step response waveform.
    """

    ntwk = rf.Network(filename)

    # Form frequency vector.
    f = ntwk.f
    # fmin = f[0]
    # if(fmin == 0):  # Correct, if d.c. point was included in original data.
    #     fmin = f[1]
    fmin = f_step
    fmax = f[-1]
    f = np.arange(fmin, fmax + fmin, fmin)
    F = rf.Frequency.from_f(f / 1e9)  # skrf.Frequency.from_f() expects its argument to be in units of GHz.

    # Form impulse response from frequency response.
    H = sdd_21(ntwk).interpolate_from_f(F).s[:, 0, 0]
    # ntwk = ntwk.interpolate_from_f(F)
    # H = np.concatenate((H, np.conj(np.flipud(H[:-1]))))  # Forming the vector that fft() would've outputted.
    H = np.pad(H, (1, 0), "constant", constant_values=1.0)  # Presume d.c. value = 1.
    if windowed:
        window = get_window(6.0, 2 * len(H))[len(H) :]
        H *= window
    # h = np.real(np.fft.ifft(H))
    if padded:
        h = np.fft.irfft(H, int(1.0 / (fmin * sample_per)) + 1)
        fmax = 1.0 / (2.0 * sample_per)
    else:
        h = np.fft.irfft(H)
    h /= np.abs(h.sum())  # Equivalent to assuming that step response settles at 1.

    # Form step response from impulse response.
    s = np.cumsum(h)

    # Form time vector.
    t0 = 1.0 / (2.0 * fmax)  # Sampling interval = 1 / (2 fNyquist).
    t = np.array([n * t0 for n in range(len(h))])

    return interp_time(t, s, sample_per)


def lfsr_bits(taps, seed):
    """
    Given a set of tap indices and a seed, generate a PRBS.

    Args:
        taps([int]): The set of fed back taps.
                     (Largest determines order of generator.)
        seed(int): The initial value of the shift register.

    Returns:
        A PRBS generator object with a next() method, for retrieving
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
    """
    Determines the center of the pulse response, using the "Hula Hoop"
    algorithm (See SiSoft/Tellian's DesignCon 2016 paper.)

    Args:
        p([Float]): The single bit pulse response.
        nspui(Int): The number of vector elements per unit interval.

    Returns:
        clock_pos(Int): The estimated index at which the clock will
                        sample the main lobe.
        thresh(Float):  The vertical threshold at which the main lobe is
                        UI wide.
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

    # mod_path = package.__file__

    # fn = os.path.basename(mod_path)
    # pathname = os.path.dirname(mod_path)

    # if fn not in ("__init__.py", "__init__.pyc"):
    #     return None

    rst = {}

    for imp, name, _ in pkgutil.iter_modules(package.__path__):
        fullModuleName = "{0}.{1}".format(package.__name__, name)
        mod = importlib.import_module(fullModuleName, package=package.__path__)
        # loader = imp.find_module(name)
        # mod = loader.load_module(package.__name__ + "." + name)
        rst[name] = mod

    return rst 
