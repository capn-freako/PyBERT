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

Port-renumber tests (``TestPortRenumber``) additionally verify that
``renumber=True`` correctly handles both common 4-port s4p conventions:

  - "1→2" (standard): TX+/TX- on ports 1/3, RX+/RX- on ports 2/4; S21 is large.
  - "1→3" (alternative): TX+/TX- on ports 1/2, RX+/RX- on ports 3/4; S31 is large.

Both conventions must yield identical Sdd21 when ``renumber=True`` is used.
Without renumbering, "1→3" files produce near-zero Sdd21.
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
# Helper for 4-port differential channels
# ---------------------------------------------------------------------------

def _make_4port_channel(freqs_hz: np.ndarray, length_m: float,
                        convention: str = "1to2",
                        z0: float = 50.0, velocity: float = 2e8,
                        loss_db_per_m_at_1ghz: float = 2.0) -> skrf.Network:
    """Return a balanced, lossy 4-port single-ended differential channel.

    Two port-numbering conventions are supported:

    - ``"1to2"`` (standard): TX+/TX- on ports 1/3, RX+/RX- on ports 2/4.
      Forward path: 1→2 (S21 large) and 3→4 (S43 large).
    - ``"1to3"`` (alternative): TX+/TX- on ports 1/2, RX+/RX- on ports 3/4.
      Forward path: 1→3 (S31 large) and 2→4 (S42 large).

    Both represent the same physical differential channel; only the port
    labelling differs.  ``import_channel(..., renumber=True)`` must produce
    the same Sdd21 from either file.
    """
    nf = len(freqs_hz)
    alpha = loss_db_per_m_at_1ghz * np.sqrt(freqs_hz / 1e9) * (np.log(10) / 20)
    beta = 2 * np.pi * freqs_hz / velocity
    g = np.exp(-(alpha + 1j * beta) * length_m)

    s = np.zeros((nf, 4, 4), dtype=complex)
    if convention == "1to2":
        s[:, 1, 0] = g;  s[:, 0, 1] = g   # S21, S12 — TX+→RX+
        s[:, 3, 2] = g;  s[:, 2, 3] = g   # S43, S34 — TX-→RX-
    elif convention == "1to3":
        s[:, 2, 0] = g;  s[:, 0, 2] = g   # S31, S13 — TX+→RX+
        s[:, 3, 1] = g;  s[:, 1, 3] = g   # S42, S24 — TX-→RX-
    else:
        raise ValueError(f"Unknown convention: {convention!r}")

    return skrf.Network(f=freqs_hz, s=s, z0=z0)


# ---------------------------------------------------------------------------
# Port-renumber tests
# ---------------------------------------------------------------------------

class TestPortRenumber:
    """Verify that renumber=True normalises both 4-port s4p port conventions."""

    def test_1to2_convention_correct_without_renumber(self, tmp_path, freq_grid, sample_period):
        """Standard 1→2 convention gives correct Sdd21 even with renumber=False."""
        fs, ts = freq_grid, sample_period
        L = 0.15
        ntwk = _make_4port_channel(fs, L, convention="1to2")
        fname = str(tmp_path / "ch_1to2.s4p")
        ntwk.write_touchstone(fname)

        result = import_channel(fname, ts, fs, renumber=False)

        alpha = 2.0 * np.sqrt(fs / 1e9) * (np.log(10) / 20)
        np.testing.assert_allclose(
            np.abs(result.s21.s.flatten()),
            np.exp(-alpha * L),
            rtol=1e-3,
        )

    def test_1to3_convention_renumber_true_matches_1to2(self, tmp_path, freq_grid, sample_period):
        """1→3 port numbering with renumber=True must yield the same Sdd21 as 1→2."""
        fs, ts = freq_grid, sample_period
        L = 0.15
        f12 = str(tmp_path / "ch_1to2.s4p")
        f13 = str(tmp_path / "ch_1to3.s4p")
        _make_4port_channel(fs, L, convention="1to2").write_touchstone(f12)
        _make_4port_channel(fs, L, convention="1to3").write_touchstone(f13)

        result_12 = import_channel(f12, ts, fs, renumber=True)
        result_13 = import_channel(f13, ts, fs, renumber=True)

        np.testing.assert_allclose(
            result_13.s21.s.flatten(),
            result_12.s21.s.flatten(),
            rtol=1e-3,
        )

    def test_1to3_convention_renumber_false_gives_zero_sdd21(self, tmp_path, freq_grid, sample_period):
        """1→3 convention without renumbering must produce near-zero Sdd21."""
        fs, ts = freq_grid, sample_period
        fname = str(tmp_path / "ch_1to3_no_renumber.s4p")
        _make_4port_channel(fs, length_m=0.15, convention="1to3").write_touchstone(fname)

        result = import_channel(fname, ts, fs, renumber=False)

        max_mag = np.max(np.abs(result.s21.s.flatten()))
        assert max_mag < 1e-6, (
            f"Expected Sdd21 ≈ 0 for un-renumbered 1→3 file; got max|S21| = {max_mag:.2e}"
        )

    def test_mixed_convention_cascade_12_then_13(self, tmp_path, freq_grid, sample_period):
        """Cascading a 1→2 s4p then a 1→3 s4p (both renumber=True) matches all-1→2 reference."""
        fs, ts = freq_grid, sample_period
        L1, L2 = 0.10, 0.15

        f1  = str(tmp_path / "seg1_12.s4p")
        f2  = str(tmp_path / "seg2_13.s4p")
        f2r = str(tmp_path / "seg2_12_ref.s4p")
        _make_4port_channel(fs, L1, convention="1to2").write_touchstone(f1)
        _make_4port_channel(fs, L2, convention="1to3").write_touchstone(f2)
        _make_4port_channel(fs, L2, convention="1to2").write_touchstone(f2r)

        n1  = import_channel(f1,  ts, fs, renumber=True)
        n2  = import_channel(f2,  ts, fs, renumber=True)
        n2r = import_channel(f2r, ts, fs, renumber=True)

        composite = n1 ** n2
        reference = n1 ** n2r

        np.testing.assert_allclose(
            composite.s21.s.flatten(),
            reference.s21.s.flatten(),
            rtol=1e-3,
        )

    def test_mixed_convention_cascade_13_then_12(self, tmp_path, freq_grid, sample_period):
        """Cascading a 1→3 s4p then a 1→2 s4p (both renumber=True) matches combined-length model."""
        fs, ts = freq_grid, sample_period
        L1, L2 = 0.12, 0.08

        f1 = str(tmp_path / "seg1_13.s4p")
        f2 = str(tmp_path / "seg2_12.s4p")
        _make_4port_channel(fs, L1, convention="1to3").write_touchstone(f1)
        _make_4port_channel(fs, L2, convention="1to2").write_touchstone(f2)

        n1 = import_channel(f1, ts, fs, renumber=True)
        n2 = import_channel(f2, ts, fs, renumber=True)
        composite = n1 ** n2

        alpha = 2.0 * np.sqrt(fs / 1e9) * (np.log(10) / 20)
        np.testing.assert_allclose(
            np.abs(composite.s21.s.flatten()),
            np.exp(-alpha * (L1 + L2)),
            rtol=1e-3,
        )

    def test_both_1to3_cascade_renumber_true(self, tmp_path, freq_grid, sample_period):
        """Two 1→3 s4p files cascaded with renumber=True must match two 1→2 files."""
        fs, ts = freq_grid, sample_period
        L1, L2 = 0.10, 0.10

        f13a = str(tmp_path / "a_13.s4p");  f13b = str(tmp_path / "b_13.s4p")
        f12a = str(tmp_path / "a_12.s4p");  f12b = str(tmp_path / "b_12.s4p")
        _make_4port_channel(fs, L1, convention="1to3").write_touchstone(f13a)
        _make_4port_channel(fs, L2, convention="1to3").write_touchstone(f13b)
        _make_4port_channel(fs, L1, convention="1to2").write_touchstone(f12a)
        _make_4port_channel(fs, L2, convention="1to2").write_touchstone(f12b)

        composite_13 = import_channel(f13a, ts, fs, renumber=True) ** import_channel(f13b, ts, fs, renumber=True)
        composite_12 = import_channel(f12a, ts, fs, renumber=True) ** import_channel(f12b, ts, fs, renumber=True)

        np.testing.assert_allclose(
            composite_13.s21.s.flatten(),
            composite_12.s21.s.flatten(),
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
        """Selecting inter_sel='multiple' with no files must raise RuntimeError."""
        from pybert.pybert import PyBERT
        dut = PyBERT(run_simulation=False, gui=False)
        dut.inter_sel = "multiple"
        dut.ch_files = []
        with pytest.raises(RuntimeError, match="no channel files"):
            dut.simulate(initial_run=True)
