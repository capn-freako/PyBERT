from typing import Any, TypeAlias, TypeVar
import numpy.typing as npt  # type: ignore

Real = TypeVar('Real', float, float)
Comp = TypeVar('Comp', complex, complex)
Rvec: TypeAlias = npt.NDArray["Real"]
Cvec: TypeAlias = npt.NDArray["Comp"]
PI: float
TWOPI: float
TestConfig = tuple[str, tuple[dict[str, Any], dict[str, Any]]]
TestSweep = tuple[str, str, list[TestConfig]]

def deconv_same(y: Rvec, x: Rvec) -> Rvec: ...
