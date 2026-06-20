"""Unit tests for pybert.utility.sigproc (self-contained functions only)."""

import numpy as np
import pytest

from pybert.utility.sigproc import moving_average, raised_cosine, resize_zero_pad, trim_impulse


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
