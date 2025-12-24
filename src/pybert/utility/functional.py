"""
Functional programming utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 24, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

from typing import Any, TypeVar

S = TypeVar('S', Any, Any)
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


def snd(xs: tuple[S, T, ...]) -> T:  # type: ignore
    """
    Python translation of Haskell ``snd()`` function.

    Args:
        xs: Tuple having second element of type: ``T``.

    Returns:
        Second element of given tuple.
    """

    return xs[1]
