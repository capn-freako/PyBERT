"""
Channel math utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

from functools import reduce
from numpy import (  # type: ignore
    append, array, cumsum, exp, log, log10,
    mean, ones, pi, roll, sqrt, where
)
from numpy.fft import fftshift  # type: ignore


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


def make_bathtub(centers, jit_pdf, min_val=0, rj=0, extrap=False):  # pylint: disable=too-many-locals
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
            Default: False

    Returns:
        ([real], (int,int)): A pair consisting of:
            - the vector of probabilities forming the bathtub curve, and
            - a pair consisting of the beginning/end indices of the extrapolated region.
    """
    half_len  = len(jit_pdf) // 2
    dt        = centers[1] - centers[0]  # Bins assumed to be uniformly spaced!
    try:
        jit_pdf_center_of_mass = int(mean([k * pk for (k, pk) in enumerate(jit_pdf)]))
    except Exception as err:  # pylint: disable=broad-exception-caught
        print(f"Error finding jitter PDF center of mass: {err}", flush=True)
        jit_pdf_center_of_mass = half_len
    _jit_pdf = roll(jit_pdf, half_len - jit_pdf_center_of_mass)
    zero_locs = where(fftshift(_jit_pdf) == 0)[0]
    ext_first = 0
    ext_last  = len(jit_pdf)
    if (extrap and len(zero_locs)):
        ext_first = min(zero_locs)
        ext_last  = max(zero_locs)
        if ext_first < half_len < ext_last:
            sqrt_2pi = sqrt(2 * pi)
            ix_r = ext_first + half_len - 1
            mu_r = centers[ix_r] - sqrt(2) * rj * sqrt(-log(rj * sqrt_2pi * jit_pdf[ix_r]))
            ix_l = ext_last - half_len + 1
            mu_l = centers[ix_l] + sqrt(2) * rj * sqrt(-log(rj * sqrt_2pi * jit_pdf[ix_l]))
            jit_pdf = append(append(gaus_pdf(centers[:ix_l], mu_l, rj),
                                    jit_pdf[ix_l: ix_r + 1]),
                             gaus_pdf(centers[ix_r + 1:], mu_r, rj))
    bathtub  = list(cumsum(jit_pdf[-1: -(half_len + 1): -1]))
    bathtub.reverse()
    bathtub  = array(bathtub + list(cumsum(jit_pdf[: half_len + 1]))) * 2 * dt
    bathtub  = where(bathtub < min_val, min_val * ones(len(bathtub)), bathtub)
    return (bathtub, (ext_first, ext_last))


def gaus_pdf(x, mu, sigma):
    """
    Gaussian probability density function.
    """
    sqrt_2pi = sqrt(2 * pi)
    return exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt_2pi)
