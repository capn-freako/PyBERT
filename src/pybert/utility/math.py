"""
Channel math utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

from functools import reduce
from typing import Any, Iterator, TypeVar

from numpy import (  # type: ignore
    append, array, cumsum, exp, log10,
    maximum, ones, pi, sqrt, where
)
from numpy.fft import fftshift  # type: ignore

from pybert.utility.sigproc import moving_average

from ..common import Rvec

T = TypeVar('T', Any, Any)


def lfsr_bits(taps: list[int], seed: int) -> Iterator[int]:
    """
    Given a set of tap indices and a seed, generate a PRBS.

    Args:
        taps: The set of fed back taps.
            (Largest determines order of generator.)
        seed: The initial value of the shift register.

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
    "Guards against pesky 'Divide by 0' error messages."

    if hasattr(x, "__len__"):
        x = where(x == 0, 1.0e-20 * ones(len(x)), x)
    else:
        if x == 0:
            x = 1.0e-20

    return log10(x)


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def make_bathtub(centers: Rvec, jit_pdf: Rvec, min_val: float = 0,
                 rj: float = 0, mu_r: float = 0, mu_l: float = 0,
                 extrap: bool = False) -> tuple[Rvec, tuple[int, int]]:
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
        bathtub: the vector of probabilities forming the bathtub curve
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
        if extrap:
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
