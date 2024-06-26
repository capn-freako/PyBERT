from pyibisami.common import *
from _typeshed import Incomplete
from typing import Any, Iterator, Optional

def loadWave(filename): ...
def interpFile(filename, sample_per): ...

class AMIModelInitializer:
    ami_params: Incomplete
    info_params: Incomplete
    def __init__(self, ami_params: dict, info_params: Optional[dict] = ..., **optional_args) -> None: ...
    channel_response: Incomplete
    row_size: Incomplete
    num_aggressors: Incomplete
    sample_interval: Incomplete
    bit_time: Incomplete

class AMIModel:
    def __init__(self, filename: str) -> None: ...
    def __del__(self) -> None: ...
    def initialize(self, init_object: AMIModelInitializer): ...
    def getWave(self, wave: Rvec, bits_per_call: int = ...) -> tuple[Rvec, Rvec, list[str]]: ...
    def get_responses(self, bits_per_call: int = ..., bit_gen: Optional[Iterator[int]] = ...) -> dict[str, Any]: ...
    initOut: Incomplete
    channel_response: Incomplete
    row_size: Incomplete
    num_aggressors: Incomplete
    sample_interval: Incomplete
    bit_time: Incomplete
    ami_params_in: Incomplete
    ami_params_out: Incomplete
    msg: Incomplete
    clock_times: Incomplete
    info_params: Incomplete
