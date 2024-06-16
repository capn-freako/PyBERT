"""
Channel modeling utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

from numpy import array, pi, power, sqrt  # type: ignore


def calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, ws):  # pylint: disable=too-many-arguments
    """Calculates propagation constant from cross-sectional parameters.

    The formula's applied are taken from Howard Johnson's "Metallic Transmission Model"
    (See "High Speed Signal Propagation", Sec. 3.1.)

    Inputs:
      - R0          skin effect resistance (Ohms/m)
      - w0          cross-over freq.
      - Rdc         d.c. resistance (Ohms/m)
      - Z0          characteristic impedance in LC region (Ohms)
      - v0          propagation velocity (m/s)
      - Theta0      loss tangent
      - ws          frequency sample points vector

    Outputs:
      - gamma       frequency dependent propagation constant
      - Zc          frequency dependent characteristic impedance
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    Rac = R0 * sqrt(2 * 1j * w / w0)  # AC resistance vector
    R = sqrt(power(Rdc, 2) + power(Rac, 2))  # total resistance vector
    L0 = Z0 / v0  # "external" inductance per unit length (H/m)
    C0 = 1.0 / (Z0 * v0)  # nominal capacitance per unit length (F/m)
    C = C0 * power((1j * w / w0), (-2.0 * Theta0 / pi))  # complex capacitance per unit length (F/m)
    gamma = sqrt((1j * w * L0 + R) * (1j * w * C))  # propagation constant (nepers/m)
    Zc = sqrt((1j * w * L0 + R) / (1j * w * C))  # characteristic impedance (Ohms)
    Zc[0] = Z0  # d.c. impedance blows up and requires correcting.

    return (gamma, Zc)


def calc_gamma_RLGC(R, L, G, C, ws):
    """Calculates propagation constant from R, L, G, and C.

    Inputs:
      - R           resistance per unit length (Ohms/m)
      - L           inductance per unit length (Henrys/m)
      - G           conductance per unit length (Siemens/m)
      - C           capacitance per unit length (Farads/m)
      - ws          frequency sample points vector

    Outputs:
      - gamma       frequency dependent propagation constant
      - Zc          frequency dependent characteristic impedance
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12

    gamma = sqrt((1j * w * L + R) * (1j * w * C + G))  # propagation constant (nepers/m)
    Zc = sqrt((1j * w * L + R) / (1j * w * C + G))  # characteristic impedance (Ohms)

    return (gamma, Zc)


def calc_G(H, Rs, Cs, Zc, RL, Cp, ws):  # pylint: disable=too-many-arguments
    """Calculates fully loaded transfer function of complete channel.

    Inputs:
      - H     unloaded transfer function of interconnect
      - Rs    source series resistance (differential)
      - Cs    source parallel (parasitic) capacitance (single ended)
      - Zc    frequency dependent characteristic impedance of the interconnect
      - RL    load resistance (differential)
      - Cp    load parallel (parasitic) capacitance (single ended)
      - ws    frequency sample points vector

    Outputs:
      - G     transfer function of fully loaded channel
    """

    w = array(ws).copy()

    # Guard against /0.
    if w[0] == 0:
        w[0] = 1.0e-12
    if Cp == 0:
        Cp = 1e-18

    def Rpar2C(R, C):
        """Calculates the impedance of the parallel combination of `R` with two
        `C`s in series."""
        return R / (1.0 + 1j * w * R * C / 2)

    # Impedance looking back into the Tx output is a simple parallel RC network.
    Zs = Rpar2C(Rs, Cs)  # The parasitic capacitances are in series.
    # Rx load impedance is parallel comb. of Rterm & parasitic cap.
    # (The two parasitic capacitances are in series.)
    ZL = Rpar2C(RL, Cp)
    # Admittance into the interconnect is (Cs || Zc) / (Rs + (Cs || Zc)).
    Cs_par_Zc = Rpar2C(Zc, Cs)
    Y = Cs_par_Zc / (Rs + Cs_par_Zc)
    # Reflection coefficient at Rx:
    R1 = (ZL - Zc) / (ZL + Zc)
    # Reflection coefficient at Tx:
    R2 = (Zs - Zc) / (Zs + Zc)
    # Fully loaded channel transfer function:
    return Y * H * (1 + R1) / (1 - R1 * R2 * H**2)
