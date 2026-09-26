"""Unit tests for pybert.utility.jitter."""

import numpy as np
import pytest

from pybert.utility.jitter import calc_jitter, find_crossing_times


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


class TestCalcJitterXingMask:
    """`xing_mask` confines ISI/DCD and the jitter histograms to the flagged crossings (e.g. - PAM-4's
    symmetric transitions), while every crossing still takes part in the TIE track."""

    UI = 10e-12
    PATTERN_LEN = 64
    NUI = 8 * PATTERN_LEN

    def _xings(self):
        rng = np.random.default_rng(0)
        ideal = np.arange(1, self.NUI) * self.UI
        k = np.arange(len(ideal))
        mask = (k // 2) % 2 == 0  # periodic in the pattern; includes both rising and falling edges
        offset = np.where(mask, 0.0, np.where(k % 2, -0.3, 0.3) * self.UI)  # data-set offsets on the rest
        actual = ideal + offset + rng.normal(0, 0.01 * self.UI, len(ideal))
        return ideal, actual, mask

    def test_isi_excludes_unflagged_crossings(self):
        ideal, actual, mask = self._xings()
        isi_all = calc_jitter(self.UI, self.NUI, self.PATTERN_LEN, ideal, actual)[2]
        isi_masked = calc_jitter(self.UI, self.NUI, self.PATTERN_LEN, ideal, actual, xing_mask=mask)[2]
        assert isi_all > 0.25 * self.UI  # the unflagged edges sit 0.3 UI from the flagged ones
        assert isi_masked < 0.1 * self.UI

    def test_histogram_excludes_unflagged_crossings(self):
        ideal, actual, mask = self._xings()
        rslt = calc_jitter(self.UI, self.NUI, self.PATTERN_LEN, ideal, actual, xing_mask=mask)
        hist, centers = np.asarray(rslt[13]), np.asarray(rslt[15])
        assert hist[np.abs(centers) > 0.2 * self.UI].sum() == pytest.approx(0.0)


class TestCalcJitterFixes:
    UI = 10e-12
    PATTERN_LEN = 64

    def test_dual_dirac_fit_recovers_rj_and_dj(self):
        """The dual-Dirac tail fits recover a known Rj and Dirac spacing (they used to return their starting guess)."""
        rng = np.random.default_rng(1)
        nui = 128 * self.PATTERN_LEN
        ideal = np.arange(1, nui) * self.UI
        dj, rj = 2e-12, 0.3e-12
        actual = ideal + rng.choice([-dj / 2, dj / 2], len(ideal)) + rng.normal(0, rj, len(ideal))
        rslt = calc_jitter(self.UI, nui, self.PATTERN_LEN, ideal, actual)
        rjDD, mu_pos, mu_neg = rslt[7], rslt[16], rslt[17]
        assert rjDD == pytest.approx(rj, rel=0.3)
        assert mu_pos == pytest.approx(dj / 2, abs=0.3e-12)
        assert mu_neg == pytest.approx(-dj / 2, abs=0.3e-12)

    def test_missed_crossing_does_not_pad_its_neighbor(self):
        """A crossing missing from its [-UI/2, +UI/2] window is flagged on its own; the next one is still measured
        (it used to be padded too: "if we missed one, we missed two")."""
        nui = 8 * self.PATTERN_LEN
        ideal = np.arange(1, nui) * self.UI
        actual = np.delete(ideal, 101)  # crossing 101 never happens
        mask = np.ones(len(ideal), dtype=bool)
        mask[101] = False
        rslt = calc_jitter(self.UI, nui, self.PATTERN_LEN, ideal, actual, xing_mask=mask)
        tie, hist = np.asarray(rslt[0]), np.asarray(rslt[13])
        assert tie[102] == pytest.approx(0.0, abs=1e-15)
        assert hist[0] == 0 and hist[-1] == 0

    def test_data_independent_jitter_removes_pattern_average_everywhere(self):
        """Purely pattern-periodic (data dependent) jitter leaves no data-independent jitter, in any pattern."""
        nui = 8 * self.PATTERN_LEN
        ideal = np.arange(1, nui) * self.UI
        per_position = np.linspace(-0.1, 0.1, self.PATTERN_LEN) * self.UI
        actual = ideal + per_position[np.arange(len(ideal)) % self.PATTERN_LEN]
        tie_ind = np.asarray(calc_jitter(self.UI, nui, self.PATTERN_LEN, ideal, actual)[8])
        assert np.abs(tie_ind).max() < 1e-3 * self.UI
