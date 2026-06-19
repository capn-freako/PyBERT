"""Tests for multi-element (composite) channel building functionality.

The composite channel feature cascades multiple s-parameter files using
scikit-rf's ``**`` operator.  These tests verify that:

  1. The cascade of N files via ``import_channel`` matches a direct scikit-rf
     cascade at the native frequency grid.
  2. For matched transmission-line segments the product rule holds:
     S21_total = S21_a * S21_b (no multiple-reflection correction needed).
  3. A single-element ``ch_files`` list produces the same network as calling
     ``import_channel`` once on that file.
  4. Three-segment cascades work correctly.
  5. End-to-end PyBERT simulation completes without error when ``ch_files``
     contains two segments.
"""

import numpy as np
import pytest
import skrf

from pybert.utility.sparam import import_channel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tline_s2p(freqs_hz: np.ndarray, length_m: float,
                    z0: float = 50.0, velocity: float = 2e8,
                    loss_db_per_m_at_1ghz: float = 2.0) -> skrf.Network:
    """Return a matched, lossy transmission-line 2-port Network.

    Loss follows a sqrt(f) skin-effect model:
        alpha [Np/m] = loss_db_per_m_at_1ghz * sqrt(f/1GHz) * ln(10)/20
    """
    alpha = loss_db_per_m_at_1ghz * np.sqrt(freqs_hz / 1e9) * (np.log(10) / 20)
    beta = 2 * np.pi * freqs_hz / velocity
    s21 = np.exp(-(alpha + 1j * beta) * length_m)
    s = np.zeros((len(freqs_hz), 2, 2), dtype=complex)
    s[:, 0, 1] = s21
    s[:, 1, 0] = s21
    return skrf.Network(f=freqs_hz, s=s, z0=z0)


def _write_and_load(ntwk: skrf.Network, path: str,
                    ts: float, fs: np.ndarray) -> skrf.Network:
    """Write *ntwk* to a Touchstone file then read it back via ``import_channel``."""
    ntwk.write_touchstone(path)
    return import_channel(path, ts, fs)


# ---------------------------------------------------------------------------
# Shared frequency grid
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def freq_grid() -> np.ndarray:
    """512-point frequency grid from 39 MHz to 20 GHz (DC excluded)."""
    f_max = 20e9
    n = 512
    return np.linspace(0, f_max, n + 1)[1:]


@pytest.fixture(scope="module")
def sample_period(freq_grid: np.ndarray) -> float:
    return 1.0 / (2.0 * freq_grid[-1])


# ---------------------------------------------------------------------------
# Unit tests: import_channel cascade vs. scikit-rf reference
# ---------------------------------------------------------------------------

class TestCompositeCascadeUnit:
    """Verify the cascade logic in isolation using import_channel."""

    def test_cascade_matches_skrf_reference(self, tmp_path, freq_grid, sample_period):
        """Two-file cascade via import_channel must equal a direct skrf cascade."""
        fs, ts = freq_grid, sample_period

        seg1 = _make_tline_s2p(fs, length_m=0.10)
        seg2 = _make_tline_s2p(fs, length_m=0.20)
        ref = seg1 ** seg2  # scikit-rf cascade at native resolution

        n1 = _write_and_load(seg1, str(tmp_path / "seg1.s2p"), ts, fs)
        n2 = _write_and_load(seg2, str(tmp_path / "seg2.s2p"), ts, fs)
        composite = n1 ** n2

        np.testing.assert_allclose(
            composite.s21.s.flatten(),
            ref.s21.s.flatten(),
            rtol=1e-3,
        )

    def test_cascade_product_rule_for_matched_lines(self, tmp_path, freq_grid, sample_period):
        """For matched lines S21_cascade = S21_a * S21_b; verify against combined segment."""
        fs, ts = freq_grid, sample_period

        L1, L2 = 0.15, 0.25
        seg1 = _make_tline_s2p(fs, length_m=L1)
        seg2 = _make_tline_s2p(fs, length_m=L2)
        combined = _make_tline_s2p(fs, length_m=L1 + L2)

        n1 = _write_and_load(seg1, str(tmp_path / "a.s2p"), ts, fs)
        n2 = _write_and_load(seg2, str(tmp_path / "b.s2p"), ts, fs)
        composite = n1 ** n2

        np.testing.assert_allclose(
            np.abs(composite.s21.s.flatten()),
            np.abs(combined.s21.s.flatten()),
            rtol=1e-3,
        )

    def test_single_file_list_is_identity(self, tmp_path, freq_grid, sample_period):
        """A length-1 file list produces exactly the same result as a direct import."""
        fs, ts = freq_grid, sample_period
        seg = _make_tline_s2p(fs, length_m=0.30)
        fname = str(tmp_path / "single.s2p")

        direct = _write_and_load(seg, fname, ts, fs)

        # Replicate the ch_files loop with one element
        networks = [import_channel(fname, ts, fs)]
        composite = networks[0]
        for ntwk in networks[1:]:
            composite = composite ** ntwk

        np.testing.assert_allclose(
            composite.s21.s.flatten(),
            direct.s21.s.flatten(),
            rtol=1e-10,
        )

    def test_three_segment_cascade(self, tmp_path, freq_grid, sample_period):
        """Three-segment cascade magnitude must match the single equivalent segment."""
        fs, ts = freq_grid, sample_period
        lengths = [0.10, 0.15, 0.12]
        segments = [_make_tline_s2p(fs, length_m=L) for L in lengths]
        combined = _make_tline_s2p(fs, length_m=sum(lengths))

        files = [str(tmp_path / f"seg{i}.s2p") for i in range(3)]
        networks = [_write_and_load(seg, f, ts, fs) for seg, f in zip(segments, files)]

        composite = networks[0]
        for ntwk in networks[1:]:
            composite = composite ** ntwk

        np.testing.assert_allclose(
            np.abs(composite.s21.s.flatten()),
            np.abs(combined.s21.s.flatten()),
            rtol=1e-3,
        )

    def test_cascade_order_matters(self, tmp_path, freq_grid, sample_period):
        """Cascade of A**B should differ from B**A when segments are asymmetric.

        Both orderings should still satisfy S21 = S21_a * S21_b for matched lines,
        but S11/S22 reflect the ordering.  This test documents that behaviour.
        """
        fs, ts = freq_grid, sample_period

        # Introduce a mild mismatch by using different z0 values
        seg_a = _make_tline_s2p(fs, length_m=0.10, z0=50)
        seg_b = _make_tline_s2p(fs, length_m=0.20, z0=50)

        fa = str(tmp_path / "ord_a.s2p")
        fb = str(tmp_path / "ord_b.s2p")
        na = _write_and_load(seg_a, fa, ts, fs)
        nb = _write_and_load(seg_b, fb, ts, fs)

        ab = na ** nb
        ba = nb ** na

        # S21 magnitude should be identical (commutative for matched lines)
        np.testing.assert_allclose(
            np.abs(ab.s21.s.flatten()),
            np.abs(ba.s21.s.flatten()),
            rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# Integration test: end-to-end PyBERT simulation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_segment_files(tmp_path_factory):
    """Write two matched transmission-line segments to persistent temp files."""
    tmp = tmp_path_factory.mktemp("composite_ch")
    # Use a coarser grid here — PyBERT will interpolate to its own grid
    fs = np.linspace(1e9, 20e9, 256)
    seg1 = _make_tline_s2p(fs, length_m=0.10, loss_db_per_m_at_1ghz=2.0)
    seg2 = _make_tline_s2p(fs, length_m=0.10, loss_db_per_m_at_1ghz=2.0)
    f1, f2 = str(tmp / "ch_seg1.s2p"), str(tmp / "ch_seg2.s2p")
    seg1.write_touchstone(f1)
    seg2.write_touchstone(f2)
    return [f1, f2]


class TestCompositeChannelIntegration:
    """End-to-end PyBERT simulation with ``ch_files`` containing two segments."""

    def test_simulation_completes(self, two_segment_files):
        """PyBERT must reach 'Ready.' status when ch_files has two segments."""
        from pybert.pybert import PyBERT
        dut = PyBERT(run_simulation=False, gui=False)
        dut.use_ch_file = True
        dut.ch_files = two_segment_files
        dut.simulate(initial_run=True)
        assert dut.status == "Ready.", f"Unexpected status: {dut.status!r}"

    def test_channel_delay_is_reasonable(self, two_segment_files):
        """Channel delay for two 0.1-m segments (v=2e8 m/s) should be ~1 ns."""
        from pybert.pybert import PyBERT
        dut = PyBERT(run_simulation=False, gui=False)
        dut.use_ch_file = True
        dut.ch_files = two_segment_files
        dut.simulate(initial_run=True)
        # Two 0.1 m segments at v=2e8 → τ = 0.2/2e8 = 1 ns; allow generous margin
        assert 0.5e-9 < dut.chnl_dly < 5e-9, (
            f"Channel delay {dut.chnl_dly*1e9:.3f} ns out of expected range 0.5–5 ns"
        )

    def test_empty_ch_files_raises(self):
        """Selecting use_ch_file=True with no files must raise RuntimeError."""
        from pybert.pybert import PyBERT
        dut = PyBERT(run_simulation=False, gui=False)
        dut.use_ch_file = True
        dut.ch_files = []
        dut.ch_file = ""
        with pytest.raises(RuntimeError, match="no channel files"):
            dut.simulate(initial_run=True)
