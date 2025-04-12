"""
General signal processing utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

import re

from typing import Optional

from numpy import (  # type: ignore
    arange, argmax, array, convolve, cos, cumsum, diff,
    mean, ones, pad, pi, roll, sign, where, zeros
)
from numpy.fft import rfft  # type: ignore
from numpy.typing import NDArray  # type: ignore
from scipy.interpolate import interp1d
from scipy.signal      import freqs, invres

from ..common import Rvec, Cvec


def moving_average(a: Rvec, n: int = 3) -> Rvec:
    """
    Calculates a sliding average over the input vector.

    Uses a weighted averaging kernel, to preserve singularity
    of peak location in input data.

    Args:
        a: Input vector to be averaged.

    Keyword Args:
        n: Width of averaging window, in vector samples.
            Odd numbers work best.
            Default: 3

    Returns:
        rslt: the moving average of the input vector, leaving the input vector unchanged.

    Notes:
        1. The odd code is intended to "protect" the first/last elements
           of the input vector from the averaging process.
           In PyBERT those elements "collect" the missed edges when
           assembling the TIE.
           Because of this non-standard use, those bins shouldn't be
           included in averaging.
    """
    win = ones((n + 1) // 2)
    krnl = convolve(win, win)
    krnl = krnl / krnl.sum()
    res  = convolve(a[1:-1], krnl, mode='same')
    return array([a[0]] + list(res) + [a[-1]])


def interp_time(ts: Rvec, xs: Rvec, sample_per: float) -> Rvec:
    """
    Resample time domain data, using linear interpolation.

    Args:
        ts: Original time values.
        xs: Original signal values.
        sample_per: System sample period (ts).

    Returns:
        rslt: Resampled waveform.
    """
    krnl = interp1d(ts, xs, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
    return krnl(arange(0, ts[-1], sample_per))


def import_time(filename: str, sample_per: float) -> Rvec:
    """
    Read in a time domain waveform file, resampling as appropriate, via
    linear interpolation.

    Args:
        filename: Name of waveform file to read in.
        sample_per: New sample interval

    Returns:
        rslt: Resampled waveform.
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
            except Exception:  # pylint: disable=broad-exception-caught
                continue

    return interp_time(array(ts), array(xs), sample_per)


def pulse_center(p: Rvec, nspui: int) -> tuple[int, float]:
    """
    Determines the center of the pulse response, using the "Hula Hoop"
    algorithm (See SiSoft/Tellian's DesignCon 2016 paper.)

    Args:
        p: The single bit pulse response.
        nspui: The number of vector elements per unit interval.

    Returns:
        (clock_pos, thresh): The estimated index at which the clock will
            sample the main lobe, and
            the vertical threshold at which the main lobe is one UI wide.
    """
    div = 2.0
    p_max = p.max()
    thresh = p_max / div
    main_lobe_ixs = where(p > thresh)[0]
    if not main_lobe_ixs.size:  # Sometimes, the optimizer really whacks out.
        return (-1, 0)          # Flag this, by returning an impossible index.

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


def raised_cosine(x: Cvec) -> Cvec:
    "Apply raised cosine window to input."
    len_x = len(x)
    w = (array([cos(pi * n / len_x) for n in range(len_x)]) + 1) / 2
    return w * x


# pylint: disable=too-many-locals
def calc_resps(t: Rvec, h: Rvec, ui: float, f: Rvec,  # noqa: F405
               eps: float = 1e-18) -> tuple[Rvec, Rvec, Cvec]:  # noqa: F405
    """
    From a uniformly sampled impulse response,
    calculate the: step, pulse, and frequency responses.

    Args:
        t: Time vector associated with ``h`` (s).
        h: Impulse response (V/sample).
        ui: Unit interval (s).
        f: Frequency vector associated w/ `H` (Hz).

    Keyword Args:
        eps: Threshold for floating point equality.
            Default: 1e-18

    Returns:
        s, p, H: tuple consisting of: step, pulse, and frequency responses.

    Raises:
        ValueError: If any of the following are true:
            - `t` is not uniformly spaced.
            - length of `t` is not at least length of `h`.
            - `f` is not uniformly spaced.
            - `f` does not begin with zero.
    Notes:
        1. ``t`` is assumed to be uniformly spaced and monotonic.
            (It is *not* assumed to begin at zero.)
    """
    ddt = diff(diff(t))
    if any(ddt > eps):
        raise ValueError(f"`t` must be uniformly spaced! (Largest spacing difference: {max(ddt)})")
    if len(t) < len(h):
        raise ValueError(f"Length of `t` ({len(t)}) must be at least length of `h` ({len(h)})!")
    ddf = diff(diff(f))
    if any(ddf > eps):
        raise ValueError(f"`f` must be uniformly spaced! (Largest spacing difference: {max(ddf)})")
    if f[0] != 0:
        raise ValueError(f"`f` must begin at zero! (f[0] = {f[0]})")

    s = h.cumsum()
    ts = t[1] - t[0]
    nspui = int(ui / ts)
    p = s - pad(s[:-nspui], (nspui, 0), mode="constant", constant_values=0)

    tmax = 1 / f[1]
    n_samps = int(tmax / ts + 0.5)
    _h = h.copy()
    _h.resize(n_samps, refcheck=False)  # Accommodating Tox.
    H = rfft(_h)
    fmax = 0.5 / ts
    _f = arange(0, fmax + f[1], f[1])
    krnl = interp1d(_f, H, kind="linear", assume_sorted=True)
    _H = krnl(f)

    return (s, p, _H)


# pylint: disable=too-many-locals
def trim_impulse(g: Rvec, min_len: int = 0, max_len: int = 1000000, front_porch: int = 0,
                 kept_energy: float = 0.999) -> tuple[Rvec, int]:
    """
    Trim impulse response, for more useful display, by:

        - clipping off the tail, after given portion of the total
            first derivative power has been captured, and
        - enforcing a minimum "front porch" length if requested.

    Args:
        g: Response to trim.

    KeywordArgs:
        min_len: Minimum length of returned vector.
            Default: 0
        max_len: Maximum length of returned vector.
            Default: 1000000
        front_porch: Minimum allowed "front porch" length.
            Default: 0
        kept_energy: The portion of first derivative "energy" retained.
            Default: 99.9%

    Returns:
        (trimmed_resp, start_ix): A pair consisting of:
            - the trimmed response, and
            - the index of the chosen starting position.
    """

    # Move main lobe to center, in case of any non-causality.
    len_g = len(g)
    half_len = len_g // 2
    if argmax(g) < len_g // 4:
        _g = roll(g, half_len)
    else:
        _g = g
    max_ix = argmax(_g)

    # Capture `kept_energy` of the total first derivative energy.
    diff_g = diff(_g)
    Ptot = sum(diff_g ** 2)
    half_residual_energy = 0.5 * (1 - kept_energy)
    Pbeg = half_residual_energy * Ptot
    Pend = (1 - half_residual_energy) * Ptot
    ix_beg = 0
    ix_end = 0
    P = 0
    while P < Pbeg:
        P      += diff_g[ix_beg] ** 2
        ix_beg += 1
    ix_end = ix_beg
    while P < Pend and ix_end < len_g:
        P      += diff_g[ix_end] ** 2
        ix_end += 1

    # Enforce minimum "front porch".
    if (max_ix - ix_beg) < front_porch:
        ix_beg = max(0, int(max_ix - front_porch))  # `int(...)` is for `mypy`.
    # Enforce minimum length.
    if (ix_end - ix_beg) < min_len:
        ix_end = min(len_g, ix_beg + min_len)
    # Enforce maximum length.
    if (ix_end - ix_beg) > max_len:
        ix_end = ix_beg + max_len

    return (_g[ix_beg:ix_end], ix_beg - half_len)


def make_ctle(rx_bw: float, peak_freq: float, peak_mag: float, w: Rvec) -> tuple[Rvec, Cvec]:  # pylint: disable=too-many-arguments  # noqa: F405
    """
    Generate the frequency response of a continuous time linear equalizer (CTLE), given the:

    - signal path bandwidth,
    - peaking specification, and
    - list of frequencies of interest.

    Args:
        rx_bw: The natural (or, unequalized) signal path bandwidth (Hz).
        peak_freq: The location of the desired peak in the frequency response (Hz).
        peak_mag: The desired relative magnitude of the peak (dB).
        w: The list of frequencies of interest (rads./s).

    Returns:
        w, H: The resultant complex frequency response, at the given frequencies.

    Notes:
        1. We use the 'invres()' function from scipy.signal, as it suggests
            itself as a natural approach, given our chosen use model of having
            the user provide the peaking frequency and degree of peaking.

            That is, we define our desired frequency response using one zero
            and two poles, where:

            - The pole locations are equal to:
                - the signal path natural bandwidth, and
                - the user specified peaking frequency.

            - The zero location is chosen, so as to provide the desired degree
                of peaking.
    """

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
    H /= max(abs(H))

    return (w, H)


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def calc_eye(ui: float, samps_per_ui: int, height: int, ys: Rvec, y_max: float,
             clock_times: Optional[Rvec] = None) -> NDArray:
    """
    Calculates the "eye" diagram of the input signal vector.

    Args:
        ui: unit interval (s)
        samps_per_ui: # of samples per unit interval
        height: height of output image data array
        ys: signal vector of interest
        y_max: max. +/- vertical extremity of plot

    Keyword Args:
        clock_times: Vector of clock times to use for eye centers.
            If not provided, just use mean zero-crossing and assume constant UI and no phase jumps.
            (This allows the same function to be used for eye diagram
            creation, for both pre and post-CDR signals.)
            Default: None

    Returns:
        eye: The "heat map" representing the eye diagram.
            Each grid location contains a value indicating the number
            of times the signal passed through that location.
    """

    # List/array necessities.
    ys = array(ys)

    # Intermediate variable calculation.
    tsamp = ui / samps_per_ui

    # Adjust the scaling.
    width = 2 * samps_per_ui
    y_scale = height // (2 * y_max)  # (pixels/V)
    y_offset = height // 2           # (pixels)

    # Generate the "heat" picture array.
    img_array = zeros([height, width])
    if clock_times is not None:
        for clock_time in clock_times:
            first_ix = int(clock_time // tsamp)
            if first_ix + 2 * samps_per_ui > len(ys):
                break
            for i, y in enumerate(ys[first_ix: first_ix + 2 * samps_per_ui]):
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
    else:
        start_ix = where(diff(sign(ys)))[0][0] + samps_per_ui // 2
        last_first_ix = len(ys) - 2 * samps_per_ui
        for first_ix in range(start_ix, last_first_ix, samps_per_ui):
            for i, y in enumerate(ys[first_ix: first_ix + 2 * samps_per_ui]):
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1

    return img_array


def make_uniform(t: Rvec, jitter: Rvec, ui: float, nbits: int) -> tuple[Rvec, list[int]]:
    """
    Make the jitter vector uniformly sampled in time, by zero-filling where
    necessary.

    The trick, here, is creating a uniformly sampled input vector for the FFT operation,
    since the jitter samples are almost certainly not uniformly sampled.
    We do this by simply zero padding the missing samples.

    Args:
        t: The sample times for the 'jitter' vector.
        jitter: The input jitter samples.
        ui: The nominal unit interval.
        nbits: The desired number of unit intervals, in the time domain.

    Returns:
        (y, y_vld): A pair consisting of:
            - The uniformly sampled, zero padded jitter vector.
            - The indices where y is valid (i.e. - not zero padded).
    """

    if len(t) < len(jitter):
        jitter = jitter[: len(t)]

    run_lengths = list(map(int, diff(t) / ui + 0.5))
    valid_ix = [0] + list(cumsum(run_lengths))
    valid_ix = [x for x in valid_ix if x < nbits]
    missing = where(array(run_lengths) > 1)[0]
    num_insertions = 0
    _jitter = list(jitter)  # Because we use 'insert'.

    for i in missing:
        for _ in range(run_lengths[i] - 1):
            _jitter.insert(i + 1 + num_insertions, 0.0)
            num_insertions += 1

    if len(_jitter) < nbits:
        _jitter.extend([0.0] * (nbits - len(_jitter)))
    if len(_jitter) > nbits:
        _jitter = _jitter[:nbits]

    return array(_jitter), valid_ix
