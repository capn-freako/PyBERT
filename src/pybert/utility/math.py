"""
Channel math utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>

Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

from functools import reduce
from itertools import pairwise
from collections.abc import Iterator

from numpy import (
    append, arange, array, cumsum, exp, linspace, log10,
    maximum, pi, polyfit, searchsorted, sort, sqrt, where
)
from numpy.fft import fftshift
from scipy.stats import norm

from pybert.utility.sigproc import moving_average

from ..common import Rvec, T


def lfsr_bits(taps: list[int], seed: int) -> Iterator[int]:
    """
    Given a set of tap indices and a seed, generate a PRBS.

    Args:
        taps: The set of fed back taps.
            (Largest determines order of generator.)
        seed: The initial value of the shift register.

    Returns:
        A PRBS generator object with a ``next()`` method for retrieving the next bit in the sequence.
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


def safe_log10(x: float | Rvec, min_val: float = 1e-20) -> float | Rvec:
    """
    Guards against pesky 'Divide by 0' error messages.

    Args:
        x: Value to take log10 of.

    Keyword Args:
        min_val: Minimum input value to allow.
            Default: 1e-20
    """

    try:
        return log10(maximum(x, min_val))
    except Exception as err:
        raise ValueError(f"x: {x}") from err


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def make_bathtub(centers: Rvec, jit_pdf: Rvec, min_val: float = 0,
                 rj: float = 0, mu_r: float = 0, mu_l: float = 0,
                 extrap: bool = False) -> Rvec:
    """
    Generate the "bathtub" curve associated with a particular jitter PDF.

    Args:
        centers: List of uniformly spaced bin centers (s).
            Note: First and last elements are exceptions.
        jit_pdf: PDF of jitter.

    Keyword Args:
        min_val: Minimum value allowed in returned bathtub vector.
            Default: 0
        rj: Standard deviation of Gaussian PDF characterizing random jitter.
            Default: 0
        mu_r: Mean of Gaussian PDF best fit to right tail.
            Default: 0
        mu_l: Mean of Gaussian PDF best fit to left tail.
            Default: 0
        extrap: Extrapolate bathtub tails, using `rj`, if True.
            Default: False

    Returns:
        The vector of probabilities forming the bathtub curve.

    ToDo:
        1. Log "rj == 0", so caller knows why bathtub edges are vertical.
    """

    half_len  = len(jit_pdf) // 2
    dt = centers[2] - centers[1]  # Avoiding `centers[0]`, due to its special nature.

    if jit_pdf[0] or jit_pdf[-1]:  # Closed eye?
        half_ui = centers[-1]
        # The following line works in conjunction w/ the line just before `return ...`,
        # to eliminate artifactual "spikes" near the center of the final plot,
        # which can occur in closed eye situations, due to slight mis-centering of the jitter PDF.
        jit_pmf = [(jit_pdf[0] + jit_pdf[-1]) * half_ui] + list(array(jit_pdf[1:-1]) * dt) + [0]
    else:
        if extrap and rj > 0:  # (No Rj, e.g. - too few pattern repetitions to separate it: no extrapolation.)
            # The weird scaling is meant to improve numerical precision through `gaus_pdf()`.
            gaus_fit  = append(gaus_pdf(centers[:half_len] * 1e12, mu_l * 1e12, rj * 1e12),
                               gaus_pdf(centers[half_len:] * 1e12, mu_r * 1e12, rj * 1e12)) * 1e-12
            jit_pdf_ext = moving_average(where(jit_pdf == 0, gaus_fit, jit_pdf), n=5)
        else:
            jit_pdf_ext = jit_pdf
        jit_pmf = array(jit_pdf_ext) * dt

    jit_cdf = cumsum(jit_pmf) * 2
    jit_cdf -= (jit_cdf[0] + jit_cdf[-1]) / 2 - 1       # Forcing mid-point to 1, because we're going to...
    jit_cdf[half_len:] -= 2 * (jit_cdf[half_len:] - 1)  # ...fold the second half vertically about the horizontal line: y=1.
    return maximum(min_val, fftshift(jit_cdf))


def _tail_prob(x: Rvec, v: Rvec, beyond_above: bool, n_meas: int = 10, n_fit: int = 100) -> Rvec:
    """
    Probability of a sample from `x` lying beyond each voltage in `v`
    (above it if `beyond_above`, else below it).

    Measured wherever at least `n_meas` samples lie beyond;
    elsewhere, taken from a Gaussian fit to the `n_fit` most extreme samples, made on the
    normal-probability (Q) scale, so the extrapolation follows the measured tail's own shape at its extreme.
    """
    sgn = 1 if beyond_above else -1
    xs = sort(sgn * array(x))
    vv = sgn * array(v)
    n = len(xs)
    n_beyond = n - searchsorted(xs, vv, side="right")
    p_emp = n_beyond / n

    n_fit = min(n_fit, n // 10)
    if n_fit < 3:
        return p_emp
    p_samp = (arange(n_fit, 0, -1) - 0.5) / n  # exceedance probability of each of the `n_fit` largest samples
    sigma, mu = polyfit(norm.isf(p_samp), xs[-n_fit:], 1)
    if sigma <= 0:
        return p_emp
    return where(n_beyond >= n_meas, p_emp, norm.sf((vv - mu) / sigma))


def make_vert_bathtubs(samps: Rvec, decisions: Rvec, n_bins: int = 200,
                       min_val: float = 0, min_samps: int = 20) -> list[tuple[Rvec, Rvec]]:
    """
    Generate the vertical "bathtub" curve of each eye, from slicer input samples.

    The vertical counterpart of `make_bathtub()`: each signal level's measured sample distribution,
    extended beyond the measured data by a Gaussian fit to its eye-facing tail.
    (The fit is made on the normal-probability scale, to only the most extreme samples, because residual ISI
    makes level distributions markedly sub-Gaussian in their tails: a Gaussian fitted to the histogram
    (tried, in volts) overstated them badly, even within the measured range.)

    Args:
        samps: Slicer input voltages, sampled at the recovered clock instants.
        decisions: The symbol decision made on each sample, used to assign it to a signal level.
            (Exact only when the samples contain no decision errors;
            an erroneous sample is counted as belonging to the wrong level.)

    Keyword Args:
        n_bins: Number of voltage steps spanning each eye, from one level's mean to the next's.
            Default: 200
        min_val: Minimum value allowed in the returned bathtub vectors.
            Default: 0
        min_samps: Minimum number of samples a level must have to be included.
            Default: 20

    Returns:
        One `(v, bathtub)` pair per eye, lowest eye first,
        where `bathtub[i]` is the probability of a sample from the level above the eye
        falling below `v[i]`, plus that of a sample from the level below it falling above `v[i]`.
    """
    samps = array(samps)
    decisions = array(decisions)
    levels = [samps[decisions == d] for d in sorted(set(decisions))]
    levels = [x for x in levels if len(x) >= min_samps]

    eyes = []
    for lo, hi in pairwise(levels):
        v = linspace(lo.mean(), hi.mean(), n_bins + 1)
        bathtub = _tail_prob(hi, v, beyond_above=False) + _tail_prob(lo, v, beyond_above=True)
        eyes.append((v, maximum(min_val, bathtub)))
    return eyes


def gaus_pdf(x: Rvec, mu: float, sigma: float) -> Rvec:
    "Gaussian probability density function."
    sqrt_2pi = sqrt(2 * pi)
    return exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt_2pi)


def all_combs(xss: list[list[T]]) -> list[list[T]]:
    """
    Generate all combinations of input.

    Args:
        xss: The lists of candidates for each position in the final output.

    Returns:
        All possible combinations of inputs.
    """
    if not xss:
        return [[]]
    head, *tail = xss
    yss = all_combs(tail)
    return [[x, *ys] for x in head for ys in yss]  # type: ignore
