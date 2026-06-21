"""Unit tests for pybert.utility.math."""

import numpy as np
import pytest
from scipy.integrate import quad

from pybert.utility.math import all_combs, gaus_pdf, lfsr_bits, make_bathtub, safe_log10


class TestSafeLog10:
    def test_normal_positive(self):
        assert safe_log10(100.0) == pytest.approx(2.0)

    def test_zero_clamps_to_min(self):
        result = safe_log10(0.0)
        assert result == pytest.approx(np.log10(1e-20))

    def test_negative_clamps_to_min(self):
        result = safe_log10(-5.0)
        assert result == pytest.approx(np.log10(1e-20))

    def test_array_input(self):
        x = np.array([0.0, 1.0, 10.0, 100.0])
        result = safe_log10(x)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(2.0)

    def test_custom_min_val(self):
        result = safe_log10(0.0, min_val=1e-10)
        assert result == pytest.approx(np.log10(1e-10))


class TestGausPdf:
    def test_peak_at_mean(self):
        mu, sigma = 0.0, 1.0
        xs = np.linspace(-5, 5, 1000)
        pdf = gaus_pdf(xs, mu, sigma)
        assert xs[np.argmax(pdf)] == pytest.approx(0.0, abs=0.02)

    def test_integrates_to_one(self):
        mu, sigma = 2.0, 0.5
        area, _ = quad(lambda x: gaus_pdf(x, mu, sigma), mu - 10 * sigma, mu + 10 * sigma)
        assert area == pytest.approx(1.0, rel=1e-4)

    def test_symmetric_about_mean(self):
        mu, sigma = 3.0, 1.0
        assert gaus_pdf(mu - 1, mu, sigma) == pytest.approx(gaus_pdf(mu + 1, mu, sigma))

    def test_wider_sigma_lowers_peak(self):
        assert gaus_pdf(0.0, 0.0, 2.0) < gaus_pdf(0.0, 0.0, 1.0)


class TestLfsrBits:
    def test_generates_bits(self):
        gen = lfsr_bits([7, 6], seed=1)
        bits = [next(gen) for _ in range(20)]
        assert all(b in (0, 1) for b in bits)

    def test_prbs7_period(self):
        """PRBS-7 (taps [7,6]) has period 2^7 - 1 = 127."""
        gen = lfsr_bits([7, 6], seed=1)
        bits = [next(gen) for _ in range(254)]
        assert bits[:127] == bits[127:254]  # second cycle matches first

    def test_nonzero_output(self):
        gen = lfsr_bits([7, 6], seed=1)
        bits = [next(gen) for _ in range(127)]
        assert sum(bits) > 0  # not all zeros


class TestAllCombs:
    def test_empty_input(self):
        assert all_combs([]) == [[]]

    def test_single_list(self):
        assert all_combs([[1, 2, 3]]) == [[1], [2], [3]]

    def test_two_lists(self):
        result = all_combs([[0, 1], [0, 1]])
        assert len(result) == 4
        assert [0, 0] in result
        assert [1, 1] in result

    def test_three_lists_count(self):
        result = all_combs([[0, 1]] * 3)
        assert len(result) == 8


class TestMakeBathtub:
    def test_output_length_matches_input(self):
        n = 100
        centers = np.linspace(-0.5e-9, 0.5e-9, n)
        jit_pdf = np.zeros(n)
        jit_pdf[n // 2] = 1.0 / (centers[2] - centers[1])
        result = make_bathtub(centers, jit_pdf)
        assert len(result) == n

    def test_values_nonnegative(self):
        n = 100
        centers = np.linspace(-0.5e-9, 0.5e-9, n)
        jit_pdf = np.zeros(n)
        jit_pdf[n // 2] = 1.0 / (centers[2] - centers[1])
        result = make_bathtub(centers, jit_pdf, min_val=0)
        assert np.all(result >= 0)
