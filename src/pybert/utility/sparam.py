"""
S-parameter manipulation utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

import os.path
import re

from cmath import phase, rect
from numpy import array, diff, ones, pi, where, zeros  # type: ignore
from numpy.fft import fft  # type: ignore
from skrf import Network
from skrf.network import one_port_2_two_port

from ..common import Rvec, Cvec

from .channel import calc_G
from .sigproc import import_time


def cap_mag(zs: Cvec, maxMag: float = 1.0) -> Cvec:
    """
    Cap the magnitude of a list of complex values, leaving the phase unchanged.

    Args:
        zs: The complex values to be capped.

    Keyword Args:
        maxMag: The maximum allowed magnitude.
            Default: 1

    Notes:
        1. Any pre-existing shape of the input will be preserved.
    """
    zs_flat = zs.flatten()
    subs = [rect(maxMag, phase(z)) for z in zs_flat]
    return where(abs(zs_flat) > maxMag, subs, zs_flat).reshape(zs.shape)  # pylint: disable=no-member


def mon_mag(zs: Cvec) -> Cvec:
    """Enforce monotonically decreasing magnitude in list of complex values,
    leaving the phase unchanged.

    Args:
        zs: The complex values to be adjusted.

    Notes:
        1. Any pre-existing shape of the input will be preserved.
    """
    zs_flat = zs.flatten()
    for ix in range(1, len(zs_flat)):
        zs_flat[ix] = rect(min(abs(zs_flat[ix - 1]), abs(zs_flat[ix])), phase(zs_flat[ix]))
    return zs_flat.reshape(zs.shape)


# ToDo: Are there SciKit-RF alternatives to these next two functions?  # pylint: disable=fixme
def sdd_21(ntwk: Network, renumber: bool = False) -> Network:
    """
    Given a 4-port single-ended network, return its differential 2-port network.

    Args:
        ntwk: 4-port single ended network.

    Keyword Args:
        renumber: Automatically fix "1=>3/2=>4" port numbering when True.
            Default: False

    Returns:
        Sdd: 2-port differential network.
    """
    mm = se2mm(ntwk, renumber=renumber)
    return Network(frequency=ntwk.f, s=mm.s[:, 0:2, 0:2], z0=mm.z0[:, 0:2])


def se2mm(ntwk: Network, scale: float = 0.5, renumber: bool = False) -> Network:
    """
    Given a 4-port single-ended network, return its mixed mode equivalent.

    Args:
        ntwk: 4-port single ended network.

    Keyword Args:
        scale: Normalization factor.
            Default: 0.5
        renumber: Automatically fix "1=>3/2=>4" port numbering when True.
            Default: False

    Returns:
        Smm: Mixed mode equivalent network, in the following format:
            Sdd11  Sdd12  Sdc11  Sdc12
            Sdd21  Sdd22  Sdc21  Sdc22
            Scd11  Scd12  Scc11  Scc12
            Scd21  Scd22  Scc21  Scc22
    """
    # Confirm correct network dimmensions.
    (_, rs, cs) = ntwk.s.shape
    if rs != cs:
        raise ValueError("Non-square Touchstone file S-matrix!")
    if rs != 4:
        raise ValueError("Touchstone file must have 4 ports!")

    # Detect/correct "1 => 3" port numbering.
    if renumber:
        ix = ntwk.s.shape[0] // 20  # So as not to be fooled by d.c. blocking.
        if abs(ntwk.s21.s[ix, 0, 0]) < abs(ntwk.s31.s[ix, 0, 0]):  # 1 ==> 3 port numbering?
            ntwk.renumber((1, 2), (2, 1))

    # Convert S-parameter data.
    s = zeros(ntwk.s.shape, dtype=complex)
    s[:, 0, 0] = scale * (ntwk.s11 - ntwk.s13 - ntwk.s31 + ntwk.s33).s.flatten()
    s[:, 0, 1] = scale * (ntwk.s12 - ntwk.s14 - ntwk.s32 + ntwk.s34).s.flatten()
    s[:, 0, 2] = scale * (ntwk.s11 + ntwk.s13 - ntwk.s31 - ntwk.s33).s.flatten()
    s[:, 0, 3] = scale * (ntwk.s12 + ntwk.s14 - ntwk.s32 - ntwk.s34).s.flatten()
    s[:, 1, 0] = scale * (ntwk.s21 - ntwk.s23 - ntwk.s41 + ntwk.s43).s.flatten()
    s[:, 1, 1] = scale * (ntwk.s22 - ntwk.s24 - ntwk.s42 + ntwk.s44).s.flatten()
    s[:, 1, 2] = scale * (ntwk.s21 + ntwk.s23 - ntwk.s41 - ntwk.s43).s.flatten()
    s[:, 1, 3] = scale * (ntwk.s22 + ntwk.s24 - ntwk.s42 - ntwk.s44).s.flatten()
    s[:, 2, 0] = scale * (ntwk.s11 - ntwk.s13 + ntwk.s31 - ntwk.s33).s.flatten()
    s[:, 2, 1] = scale * (ntwk.s12 - ntwk.s14 + ntwk.s32 - ntwk.s34).s.flatten()
    s[:, 2, 2] = scale * (ntwk.s11 + ntwk.s13 + ntwk.s31 + ntwk.s33).s.flatten()
    s[:, 2, 3] = scale * (ntwk.s12 + ntwk.s14 + ntwk.s32 + ntwk.s34).s.flatten()
    s[:, 3, 0] = scale * (ntwk.s21 - ntwk.s23 + ntwk.s41 - ntwk.s43).s.flatten()
    s[:, 3, 1] = scale * (ntwk.s22 - ntwk.s24 + ntwk.s42 - ntwk.s44).s.flatten()
    s[:, 3, 2] = scale * (ntwk.s21 + ntwk.s23 + ntwk.s41 + ntwk.s43).s.flatten()
    s[:, 3, 3] = scale * (ntwk.s22 + ntwk.s24 + ntwk.s42 + ntwk.s44).s.flatten()

    # Convert port impedances.
    f = ntwk.f
    z = zeros((len(f), 4), dtype=complex)
    z[:, 0] = ntwk.z0[:, 0] + ntwk.z0[:, 2]
    z[:, 1] = ntwk.z0[:, 1] + ntwk.z0[:, 3]
    z[:, 2] = (ntwk.z0[:, 0] + ntwk.z0[:, 2]) / 2
    z[:, 3] = (ntwk.z0[:, 1] + ntwk.z0[:, 3]) / 2

    return Network(frequency=f, s=s, z0=z)


def interp_s2p(ntwk: Network, f: Rvec) -> Network:
    """
    Safely interpolate a 2-port network, by applying certain constraints to
    any necessary extrapolation.

    Args:
        ntwk: The 2-port network to be interpolated.
        f: The list of new frequency sampling points (Hz).

    Returns:
        Sint: The interpolated/extrapolated 2-port network.

    Raises:
        ValueError: If `ntwk` is _not_ a 2-port network.
    """
    (_, rs, cs) = ntwk.s.shape
    if rs != cs:
        raise ValueError("Non-square Touchstone file S-matrix!")
    if rs != 2:
        raise ValueError("Touchstone file must have 2 ports!")

    extrap = ntwk.interpolate(f, fill_value="extrapolate", coords="polar", assume_sorted=True)
    if extrap.f[-1] > 1e12:
        raise ValueError(f"Maximum frequency > 1 THz!\n\tf: {f}\n\textrap: {extrap}")
    s11 = cap_mag(extrap.s[:, 0, 0])
    s22 = cap_mag(extrap.s[:, 1, 1])
    s12 = ntwk.s12.interpolate(f, fill_value=0, bounds_error=False, coords="polar", assume_sorted=True).s.flatten()
    s21 = ntwk.s21.interpolate(f, fill_value=0, bounds_error=False, coords="polar", assume_sorted=True).s.flatten()
    s = array(list(zip(zip(s11, s12), zip(s21, s22))))
    if ntwk.name is None:
        ntwk.name = "s2p"
    return Network(f=f, s=s, z0=extrap.z0, name=(ntwk.name + "_interp"), f_unit="Hz")


def H_2_s2p(H: Cvec, Zc: Cvec, fs: Rvec, Zref: float = 50) -> Network:
    """
    Convert transfer function to 2-port network.

    Args:
        H: Transfer function of medium alone.
        Zc: Complex impedance of medium.
        fs: Frequencies at which `H` and `Zc` were sampled (Hz).

    Keyword Args:
        Zref: Reference (i.e. - port) impedance to be used in constructing the network (Ohms).
            Default: 50

    Returns:
        s2p: 2-port network representing the channel to which `H` and `Zc` pertain.
    """
    # ToDo: Fix this code.  # pylint: disable=fixme
    ws = 2 * pi * fs
    G = calc_G(H, Zref, 0, Zc, Zref, 0, ws)  # See `calc_G()` docstring.
    R1 = (Zc - Zref) / (Zc + Zref)  # reflection coefficient looking into medium from port
    # T1 = 1 + R1  # transmission coefficient looking into medium from port
    Z2   = Zc * (1 - R1 * H**2)         # impedance looking into port 2, with port 1 terminated into Zref
    R2   = (Z2 - Zc) / (Z2 + Zc)      # reflection coefficient looking out of port 2
    # R2   = 0
    # Z1   = Zc * (1 + R2*H**2)         # impedance looking into port 1, with port 2 terminated into Z2
    # Calculate the one-way transfer function of medium capped w/ ports of the chosen impedance.
    # G    = calc_G(H, Zref, 0, Zc, Zc, 0, 2*pi*fs)  # See `calc_G()` docstring.
    # R2   = -R1                        # reflection coefficient looking into ref. impedance
    S21 = G
    S11  = 2 * (R1 + H * R2 * G)
    tmp = array(list(zip(zip(S11, S21), zip(S21, S11))))
    return Network(s=tmp, f=fs / 1e9, z0=[Zref, Zref])  # `f` is presumed to have units: GHz.


def import_freq(filename: str, renumber: bool = False) -> Network:
    """
    Read in a 1, 2, or 4-port Touchstone file, and return an equivalent 2-port network.

    Args:
        filename: Name of Touchstone file to read in.

    Keyword Args:
        renumber: Automatically detect/fix "1=>3/2=>4" port numbering, when True.
            Default = False

    Returns:
        s2p_DD: 2-port network.

    Raises:
        ValueError: If Touchstone file is not 1, 2, or 4-port.

    Notes:
        1. A 4-port Touchstone file is assumed single-ended,
        and the "DD" quadrant of its mixed-mode equivalent gets returned.
    """
    # Import and sanity check the Touchstone file.
    ntwk = Network(filename, f_unit="Hz")
    (_, rs, cs) = ntwk.s.shape
    if rs != cs:
        raise ValueError("Non-square Touchstone file S-matrix!")
    if rs not in (1, 2, 4):
        raise ValueError(f"Touchstone file must have 1, 2, or 4 ports!\n{ntwk}")

    # Convert to a 2-port network.
    if rs == 4:  # 4-port Touchstone files are assumed single-ended!
        return sdd_21(ntwk, renumber=renumber)
    if rs == 2:
        return ntwk
    return one_port_2_two_port(ntwk)


def import_channel(filename: str, sample_per: float, fs: Rvec,
                   zref: float = 100, renumber: bool = False) -> Network:
    """
    Read in a channel description file.

    Args:
        filename: Name of file from which to import channel description.
        sample_per: Sample period of system signal vector (s).
        fs: (Positive only) frequency values being used by caller (Hz).

    Keyword Args:
        zref: Reference impedance for time domain files (Ohms).
            Default: 100
        renumber: Automatically fix "1=>3/2=>4" port numbering when True.
            Default: False

    Returns:
        s2p: 2-port network description of channel.

    Notes:
        1. When a time domain (i.e. - impulse or step response) file is being imported,
        we have little choice but to use the given reference impedance as the channel
        characteristic impedance, for all frequencies. This implies two things:

            1. Importing time domain descriptions of channels into PyBERT
            yields necessarily lower fidelity results than importing Touchstone descriptions;
            probably not a surprise to those skilled in the art.

            2. The user should take care to ensure that the reference impedance value
            in the GUI is equal to the nominal characteristic impedance of the channel
            being imported when using time domain channel description files.
    """
    extension = os.path.splitext(filename)[1][1:]
    if re.search(r"^s\d+p$", extension, re.ASCII | re.IGNORECASE):  # Touchstone file?
        ts2N = interp_s2p(import_freq(filename, renumber=renumber), fs)
    else:  # simple 2-column time domain description (impulse or step).
        h = import_time(filename, sample_per)
        # Fixme: an a.c. coupled channel breaks this naive approach!  # pylint: disable=fixme
        if h[-1] > (max(h) / 2.0):  # step response?
            h = diff(h)  # impulse response is derivative of step response.
        Nf = len(fs)
        h.resize(2 * Nf)
        H = fft(h * sample_per)[:Nf]  # Keep the positive frequencies only.
        ts2N = H_2_s2p(H, zref * ones(len(H)), fs, Zref=zref)
    return ts2N
