"""
COM metric reporting via PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   February 29, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from pathlib import Path
from typing import Optional

from pychopmarg.com import COM
from pychopmarg.config.ieee_8023dj import IEEE_8023dj
from pychopmarg.config.template import COMParams


def calc_com(
    thru_file: str,
    fext_files: Optional[list[str]] = None,
    next_files: Optional[list[str]] = None,
    com_params: Optional[COMParams] = None,
) -> float:
    """
    Calculate the Channel Operating Margin (COM) using PyChOpMarg.

    Args:
        thru_file: Path to the thru-channel Touchstone (.s4p) file.

    Keyword Args:
        fext_files: Paths to FEXT aggressor Touchstone (.s4p) files.
            Default: None (no FEXT aggressors)
        next_files: Paths to NEXT aggressor Touchstone (.s4p) files.
            Default: None (no NEXT aggressors)
        com_params: COM configuration parameters for the desired standard.
            Default: None (uses IEEE 802.3dj parameters)

    Returns:
        COM value (dB).

    Raises:
        RuntimeError: If EQ optimization fails.
        ValueError: If channel file format is unsupported.
    """
    if com_params is None:
        com_params = IEEE_8023dj
    channels: dict[str, list[Path]] = {
        "THRU": [Path(thru_file)],
        "FEXT": [Path(f) for f in (fext_files or [])],
        "NEXT": [Path(f) for f in (next_files or [])],
    }
    com = COM(com_params, channels)
    return com()
