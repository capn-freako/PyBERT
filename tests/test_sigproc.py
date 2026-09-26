"""Unit tests for pybert.utility.sigproc (self-contained functions only)."""

from types import SimpleNamespace

import numpy as np
import pytest

from pybert.utility.sigproc import (
    add_ffe_dfe, get_dfe_weights, moving_average, raised_cosine, resize_zero_pad, trim_impulse,
)


class TestResizeZeroPad:
    def test_truncates_when_shorter(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = resize_zero_pad(x, 3)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_same_length_is_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        result = resize_zero_pad(x, 3)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_pads_end_with_zeros(self):
        x = np.array([1.0, 2.0])
        result = resize_zero_pad(x, 5)
        assert list(result) == [1.0, 2.0, 0.0, 0.0, 0.0]

    def test_pads_front_when_requested(self):
        x = np.array([1.0, 2.0])
        result = resize_zero_pad(x, 5, pad_front=True)
        assert list(result) == [0.0, 0.0, 0.0, 1.0, 2.0]

    def test_output_length_correct(self):
        x = np.ones(10)
        for n in (5, 10, 15, 20):
            assert len(resize_zero_pad(x, n)) == n


class TestMovingAverage:
    def test_dc_passthrough(self):
        # Interior elements of a DC signal pass through unchanged; only the
        # elements adjacent to the protected boundary show edge effects.
        x = np.ones(20)
        result = moving_average(x, n=5)
        assert result[3:-3] == pytest.approx(np.ones(14), rel=1e-6)

    def test_length_preserved(self):
        x = np.random.default_rng(0).random(50)
        result = moving_average(x, n=5)
        assert len(result) == len(x)

    def test_first_and_last_unchanged(self):
        x = np.array([10.0] + [1.0] * 18 + [20.0])
        result = moving_average(x, n=3)
        assert result[0] == pytest.approx(10.0)
        assert result[-1] == pytest.approx(20.0)

    def test_smooths_spike(self):
        x = np.zeros(21)
        x[10] = 100.0
        smoothed = moving_average(x, n=5)
        assert smoothed[10] < 100.0


class TestRaisedCosine:
    def test_output_length(self):
        x = np.ones(64, dtype=complex)
        result = raised_cosine(x)
        assert len(result) == 64

    def test_window_starts_at_one(self):
        # w[0] = (cos(0) + 1)/2 = 1.0, so the first element is unchanged
        x = np.ones(64, dtype=complex)
        result = raised_cosine(x)
        assert abs(result[0]) == pytest.approx(1.0, rel=1e-6)

    def test_window_tapers_to_end(self):
        # The window decreases monotonically; the last element is near 0
        x = np.ones(64, dtype=complex)
        result = raised_cosine(x)
        assert abs(result[-1]) < abs(result[0])

    def test_monotone_decreasing(self):
        x = np.ones(64, dtype=float)
        result = raised_cosine(x)
        assert np.all(np.diff(result) <= 0), "Window should be monotonically non-increasing"

    def test_real_input_gives_real_output(self):
        x = np.ones(32)
        result = raised_cosine(x)
        assert np.all(np.isreal(result))


class TestTrimImpulse:
    def test_returns_tuple(self):
        h = np.zeros(1000)
        h[100] = 1.0
        result = trim_impulse(h)
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_shorter_than_input(self):
        h = np.zeros(1000)
        h[100] = 1.0
        trimmed, _ = trim_impulse(h)
        assert len(trimmed) < len(h)

    def test_peak_preserved(self):
        h = np.zeros(1000)
        h[100] = 1.0
        trimmed, _ = trim_impulse(h)
        assert max(abs(trimmed)) == pytest.approx(1.0, rel=1e-9)

    def test_start_index_is_integer(self):
        h = np.zeros(500)
        h[200] = 1.0
        _, start_ix = trim_impulse(h)
        assert isinstance(int(start_ix), int)  # offset can be negative (relative to half-length)


NSPUI = 32
CURS = 500


def _pulse(ui_samples):
    """Single-peaked pulse, linear between UI-spaced samples; cursor (peak) at `CURS`."""
    xs = [CURS + (k - 1) * NSPUI for k in range(len(ui_samples))]
    return np.interp(np.arange(4000), xs, ui_samples)


def _tuners(*specs):
    return [SimpleNamespace(enabled=e, min_val=lo, max_val=hi) for e, lo, hi in specs]


class TestGetDfeWeights:
    def test_weights_are_isi_over_slicer_level(self):
        pr = _pulse([0.0, 2.0, 0.6, 0.2, 0.0])  # cursor 2.0; post-cursors 0.6, 0.2
        w = get_dfe_weights(_tuners((True, -1, 1), (True, -1, 1)), pr, NSPUI, decision_scaler=2.0)
        assert w == pytest.approx([0.3, 0.1])

    def test_only_leading_enabled_taps_used(self):
        pr = _pulse([0.0, 1.0, 0.3, 0.1, 0.05, 0.0])
        w = get_dfe_weights(_tuners((True, -1, 1), (False, -1, 1), (True, -1, 1)), pr, NSPUI, decision_scaler=1.0)
        assert w == pytest.approx([0.3, 0.0, 0.0])

    def test_weights_clamped_to_tuner_range(self):
        pr = _pulse([0.0, 1.0, 0.5, -0.4, 0.0])
        w = get_dfe_weights(_tuners((True, -0.2, 0.2), (True, -0.2, 0.2)), pr, NSPUI, decision_scaler=1.0)
        assert w == pytest.approx([0.2, -0.2])


class TestAddFfeDfe:
    def test_dfe_cancels_only_its_own_ui(self):
        """Each DFE tap removes its own post-cursor and nothing else -- no residual step in the tail."""
        pr = _pulse([0.0, 1.0, 0.3, 0.1, 0.0])
        out = add_ffe_dfe(np.array([]), np.array([0.3, 0.1]), np.array([1, 1]), np.array([-1, -1]), NSPUI, pr.copy(), 1.0)
        assert out[CURS] == pytest.approx(1.0)
        assert out[CURS + NSPUI] == pytest.approx(0.0)
        assert out[CURS + 2 * NSPUI] == pytest.approx(0.0)
        assert np.abs(out[CURS + 3 * NSPUI:]).max() == pytest.approx(0.0)

    def test_weights_from_get_dfe_weights_zero_the_post_cursors(self):
        """`get_dfe_weights()` and `add_ffe_dfe()` agree on units, at any slicer level."""
        pr = _pulse([0.0, 1.5, 0.45, 0.15, 0.0])
        tuners = _tuners((True, -1, 1), (True, -1, 1))
        for scaler in (1.5, 1.077):
            w = get_dfe_weights(tuners, pr, NSPUI, scaler)
            out = add_ffe_dfe(np.array([]), w, np.array([1, 1]), np.array([-1, -1]), NSPUI, pr.copy(), scaler)
            assert out[CURS + NSPUI] == pytest.approx(0.0)
            assert out[CURS + 2 * NSPUI] == pytest.approx(0.0)

    def test_dfe_timed_from_given_cursor_instant(self):
        """With `curs_ix`, DFE taps are timed from that sampling instant rather than the peak --
        e.g. MMSE's chosen instant, where its DFE weights are the equalized post-cursors."""
        pr = _pulse([0.0, 1.0, 0.3, 0.1, 0.0])
        ts = CURS - 8  # a quarter UI before the peak
        post1 = pr[ts + NSPUI]
        out = add_ffe_dfe(np.array([]), np.array([post1 / pr[ts]]), np.array([1]), np.array([-1]), NSPUI, pr.copy(),
                          pr[ts], curs_ix=ts)
        assert out[ts + NSPUI] == pytest.approx(0.0)
        assert out[ts] == pytest.approx(pr[ts])
