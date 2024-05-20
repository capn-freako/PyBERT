"""
Definitions common to all PyBERT modules.

Original author: David Banas <capn.freako@gmail.com>

Original date:   May 13, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any, Optional, TypeVar, TypeAlias  # pylint: disable=unused-import
import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore

Real = TypeVar('Real', float, float)
Comp = TypeVar('Comp', complex, complex)
Rvec: TypeAlias = npt.NDArray[Real]
Cvec: TypeAlias = npt.NDArray[Comp]

PI: float = np.pi
TWOPI: float = 2 * np.pi
