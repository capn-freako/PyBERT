"""
Functional programming utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 24, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
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


def fst(xs: tuple[T, ...]) -> T:
    """
    Python translation of Haskell ``fst()`` function.

    Args:
        xs: Tuple having first element of type: ``T``.

    Returns:
        First element of given tuple.
    """

    return xs[0]

