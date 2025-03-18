"""
Definitions common to all PyBERT modules.

Original author: David Banas <capn.freako@gmail.com>

Original date:   May 13, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any, Optional, TypeVar  # , TypeAlias  # pylint: disable=unused-import  # noqa: F401
import numpy as np  # type: ignore  # pylint: disable=unused-import  # noqa: F401
import numpy.typing as npt  # type: ignore

Real = TypeVar('Real', float, float)
Comp = TypeVar('Comp', complex, complex)
Rvec = npt.NDArray[Real]
Cvec = npt.NDArray[Comp]

PI    = np.pi
TWOPI = 2 * PI
# PI: float = 3.14159265359
# TWOPI: float = 2. * PI
