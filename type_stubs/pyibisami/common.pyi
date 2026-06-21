from typing import Any, TypeAlias, TypeVar
import numpy as np
import numpy.typing as npt  # type: ignore

Real = TypeVar('Real', np.float64, np.float64)			#: Real scalar
Comp = TypeVar('Comp', np.complex64, np.complex128)		#: Complex scalar
Rvec: TypeAlias = npt.NDArray["Real"]
Cvec: TypeAlias = npt.NDArray["Comp"]
PI: float
TWOPI: float
TestConfig = tuple[str, tuple[dict[str, Any], dict[str, Any]]]
TestSweep = tuple[str, str, list[TestConfig]]

def deconv_same(y: Rvec, x: Rvec) -> Rvec: ...
