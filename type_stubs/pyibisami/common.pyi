from typing import Any, TypeAlias, TypeVar

Real = TypeVar('Real', float, float)
Comp = TypeVar('Comp', complex, complex)
Rvec: TypeAlias
Cvec: TypeAlias
PI: float
TWOPI: float
TestConfig = tuple[str, tuple[dict[str, Any], dict[str, Any]]]
TestSweep = tuple[str, str, list[TestConfig]]

def deconv_same(y: Rvec, x: Rvec) -> Rvec: ...
