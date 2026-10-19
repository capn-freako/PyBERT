"""Unit tests for pybert.utility.math."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pybert.utility.math import all_combs, gaus_pdf, lfsr_bits, make_bathtub, make_vert_bathtubs, safe_log10


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


def _opening(v, bathtub, p):
    best = int(np.argmin(bathtub))
    lo = np.interp(p, bathtub[:best + 1][::-1], v[:best + 1][::-1])
    hi = np.interp(p, bathtub[best:], v[best:])
    return hi - lo


class TestMakeVertBathtubs:
    def test_one_eye_per_adjacent_level_pair(self):
        rng = np.random.default_rng(0)
        d = rng.choice([-1.0, -1 / 3, 1 / 3, 1.0], 4000)
        eyes = make_vert_bathtubs(d + rng.normal(0, 0.02, len(d)), d)
        assert len(eyes) == 3
        assert all(v[0] < v[-1] for v, _ in eyes)

    @pytest.mark.parametrize("ber", [1e-5, 1e-8])
    def test_gaussian_levels_extrapolate_to_analytic_opening(self, ber):
        """Levels at +/-1 with Gaussian noise (sigma = 0.1): at `ber`, each eye edge sits
        where one level's tail alone reaches `ber`, i.e. 1 - sigma * Q^-1(ber) from center.
        (Tolerance: half a sigma in total -- the fitted tail's own statistical uncertainty,
        magnified by Q^-1(ber), when extrapolating decades beyond 10k samples.)"""
        rng = np.random.default_rng(0)
        d = rng.choice([-1.0, 1.0], 10000)
        (v, bathtub), = make_vert_bathtubs(d + rng.normal(0, 0.1, len(d)), d)
        assert _opening(v, bathtub, ber) == pytest.approx(2 * (1 - 0.1 * norm.isf(ber)), abs=0.05)

    def test_bounded_tails_not_extrapolated_as_wide_gaussian(self):
        """Uniform (hard-bounded) level spread, like residual ISI: the eye stays open well
        beyond what a Gaussian with the same standard deviation would allow."""
        rng = np.random.default_rng(0)
        d = rng.choice([-1.0, 1.0], 10000)
        s = d + rng.uniform(-0.5, 0.5, len(d))  # std = 0.289; Gaussian 1e-5 opening would be < 0
        (v, bathtub), = make_vert_bathtubs(s, d)
        assert bathtub.min() < 1e-5
        assert _opening(v, bathtub, 1e-5) > 0.5
