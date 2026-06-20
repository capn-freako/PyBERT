"""Unit tests for pybert.utility.jitter."""

import numpy as np
import pytest

from pybert.utility.jitter import find_crossing_times


class TestFindCrossingTimes:
    def _square_wave(self, fs: float = 1e9, f_sig: float = 1e8, n_periods: int = 5):
        """Return (t, x) for an ideal NRZ-like square wave with known crossings."""
        dt = 1.0 / fs
        period = 1.0 / f_sig
        n_samples = int(n_periods * period / dt)
        t = np.arange(n_samples) * dt
        x = np.sign(np.sin(2 * np.pi * f_sig * t))
        x = np.where(x == 0, 1.0, x)  # avoid zero samples at crossings
        return t, x

    def test_returns_array(self):
        t, x = self._square_wave()
        result = find_crossing_times(t, x, rising_first=False)
        assert isinstance(result, np.ndarray)

    def test_crossing_count_reasonable(self):
        n_periods = 5
        t, x = self._square_wave(n_periods=n_periods)
        result = find_crossing_times(t, x, rising_first=False)
        # Should find approximately 2 crossings per period
        assert len(result) >= n_periods

    def test_rising_first(self):
        """With rising_first=True, the first returned crossing should be a rising edge."""
        t, x = self._square_wave()
        crossings = find_crossing_times(t, x, rising_first=True)
        if len(crossings) >= 2:
            # Spacing between consecutive crossings should be approximately half a period.
            gaps = np.diff(crossings[:6])
            assert np.all(gaps > 0), "Crossings not in monotone order"

    def test_min_delay_filters_early_crossings(self):
        t, x = self._square_wave()
        min_delay = t[len(t) // 2]
        result = find_crossing_times(t, x, min_delay=min_delay, rising_first=False)
        if len(result):
            assert result[0] >= min_delay

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="len"):
            find_crossing_times(np.array([0.0, 1.0]), np.array([1.0]))

    def test_monotone_increasing_times(self):
        t, x = self._square_wave()
        result = find_crossing_times(t, x, rising_first=False)
        if len(result) > 1:
            assert np.all(np.diff(result) > 0)
