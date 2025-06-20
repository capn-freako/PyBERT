"""
Definitions common to all PyBERT modules.

Original author: David Banas <capn.freako@gmail.com>

Original date:   May 13, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any, Optional, TypeAlias, TypeVar  # , TypeAlias  # pylint: disable=unused-import  # noqa: F401
import numpy as np  # type: ignore  # pylint: disable=unused-import  # noqa: F401
import numpy.typing as npt  # type: ignore

Real = TypeVar('Real', np.float64, np.float64)
Comp = TypeVar('Comp', np.complex64, np.complex128)
Rvec: TypeAlias = npt.NDArray[Real]
Cvec: TypeAlias = npt.NDArray[Comp]
Rmat: TypeAlias = npt.NDArray[Real]
Cmat: TypeAlias = npt.NDArray[Comp]

PI    = np.pi
TWOPI = 2 * PI
