"""Unit tests for pybert.utility.channel."""

import numpy as np
import pytest

from pybert.utility.channel import calc_gamma_RLGC


class TestCalcGammaRLGC:
    def _lossless_params(self):
        """Lossless line: R=0, G=0. gamma = jw*sqrt(LC), Zc = sqrt(L/C)."""
        L = 1e-9   # 1 nH/m
        C = 1e-12  # 1 pF/m
        Z0 = np.sqrt(L / C)  # 31.62 Ω
        return L, C, Z0

    def test_returns_two_arrays(self):
        L, C, Z0 = self._lossless_params()
        ws = np.linspace(1e6, 1e10, 50) * 2 * np.pi
        gamma, Zc = calc_gamma_RLGC(0.0, L, 0.0, C, ws)
        assert len(gamma) == 50
        assert len(Zc) == 50

    def test_lossless_gamma_purely_imaginary(self):
        """For R=G=0 the propagation constant is purely imaginary (no attenuation)."""
        L, C, Z0 = self._lossless_params()
        ws = np.linspace(1e8, 1e10, 100) * 2 * np.pi
        gamma, _ = calc_gamma_RLGC(0.0, L, 0.0, C, ws)
        assert np.allclose(gamma.real, 0.0, atol=1e-6)

    def test_lossless_characteristic_impedance(self):
        L, C, Z0 = self._lossless_params()
        ws = np.linspace(1e8, 1e10, 100) * 2 * np.pi
        _, Zc = calc_gamma_RLGC(0.0, L, 0.0, C, ws)
        assert np.allclose(abs(Zc), Z0, rtol=1e-4)

    def test_lossy_gamma_has_real_part(self):
        """Non-zero R should produce a real attenuation component."""
        L, C, Z0 = self._lossless_params()
        ws = np.linspace(1e8, 1e10, 50) * 2 * np.pi
        gamma, _ = calc_gamma_RLGC(10.0, L, 0.0, C, ws)
        assert np.all(gamma.real >= 0), "Attenuation must be non-negative"
        assert np.any(gamma.real > 0), "Lossy line should have positive attenuation"

    def test_dc_guard_no_exception(self):
        """DC (w=0) is guarded inside the function; should not raise."""
        L, C, _ = self._lossless_params()
        ws = np.array([0.0, 1e9, 2e9]) * 2 * np.pi
        gamma, Zc = calc_gamma_RLGC(0.0, L, 0.0, C, ws)
        assert not np.any(np.isnan(gamma))
        assert not np.any(np.isnan(Zc))
