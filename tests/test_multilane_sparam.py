"""Unit tests for 8- and 12-port Touchstone lane extraction and FEXT analysis.

Verifies that ``import_freq`` correctly extracts the requested differential
lane from multi-port single-ended S-parameter files, and that ``import_fext``
returns the correct FEXT transfer functions for aggressor lanes.

Port ordering convention (N-port, L = N/4 lanes):
    Lane k occupies ports [4k, 4k+1, 4k+2, 4k+3] (0-indexed), representing
    (+A, +B, −A, −B) respectively.  ``import_freq`` re-orders them to
    (+A, −A, +B, −B) before calling ``sdd_21()``.

FEXT path from aggressor j to victim k uses aggressor input ports (4j, 4j+2)
and victim output ports (4k+1, 4k+3), assembled as (TX+, TX−, RX+, RX−).
"""

import tempfile

import numpy as np
import pytest
import skrf

from pybert.utility.sparam import import_freq, import_fext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tline_s2p(freqs_hz: np.ndarray, loss_db: float = 3.0) -> skrf.Network:
    """Return a simple matched 2-port transmission-line network."""
    alpha = loss_db * np.sqrt(freqs_hz / 1e9) * (np.log(10) / 20)
    beta = 2 * np.pi * freqs_hz / 2e8
    s21 = np.exp(-(alpha + 1j * beta) * 0.1)
    s = np.zeros((len(freqs_hz), 2, 2), dtype=complex)
    s[:, 0, 1] = s21
    s[:, 1, 0] = s21
    return skrf.Network(f=freqs_hz, s=s, z0=50.0)


def _build_multilane_network(lane_s2p_list: list[skrf.Network]) -> skrf.Network:
    """
    Assemble L independent differential lanes into a 4L-port single-ended Network.

    Each lane k occupies four single-ended ports in (+A, +B, −A, −B) order:
        global port 4k   → +A conductor
        global port 4k+1 → +B conductor
        global port 4k+2 → −A conductor
        global port 4k+3 → −B conductor

    ``import_freq`` extracts subnetwork([4k, 4k+2, 4k+1, 4k+3]) → ports (+A, −A, +B, −B)
    then calls ``sdd_21()``, which computes:
        Sdd21 = 0.5 * (sub_s[1,0] - sub_s[1,2] - sub_s[3,0] + sub_s[3,2])
    where indices refer to (0=+A, 1=−A, 2=+B, 3=−B) in the subnetwork.

    Setting S[−B, +A] = −2·T means sub_s[3,0] = −2·T, giving Sdd21 = T exactly.
    This is the simplest SE construction that produces a known, checkable Sdd21.
    """
    L = len(lane_s2p_list)
    N = 4 * L
    n_freqs = lane_s2p_list[0].f.size
    freqs = lane_s2p_list[0].f

    s = np.zeros((n_freqs, N, N), dtype=complex)
    z0 = np.full((n_freqs, N), 50.0)

    for k, lane in enumerate(lane_s2p_list):
        # Ports for this lane: +A=4k, +B=4k+1, −A=4k+2, −B=4k+3
        p_pA, p_pB, p_mA, p_mB = 4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3
        s21 = lane.s[:, 1, 0]
        # S[−B, +A] = −2·T  →  sub_s[3, 0] = −2·T  →  Sdd21 = T
        s[:, p_mB, p_pA] = -2 * s21
        # reciprocal direction
        s[:, p_pA, p_mB] = -2 * lane.s[:, 0, 1]

    return skrf.Network(f=freqs, s=s, z0=z0)


def _build_multilane_network_with_fext(
    lane_s2p_list: list[skrf.Network],
    fext_matrix: dict[tuple[int, int], skrf.Network],
) -> skrf.Network:
    """
    Assemble L lanes and optional FEXT paths into a 4L-port single-ended Network.

    ``lane_s2p_list[k]`` is the thru-path for lane k (built as in
    ``_build_multilane_network``).

    ``fext_matrix[(j, k)]`` is the desired Sdd21 for the FEXT path from aggressor j
    to victim k.  The SE element set is ``s[4k+3, 4j] = -2·T_fext``, which makes
    the extracted Sdd21 equal to T_fext (same derivation as for the thru paths).
    """
    L = len(lane_s2p_list)
    N = 4 * L
    n_freqs = lane_s2p_list[0].f.size
    freqs = lane_s2p_list[0].f

    s = np.zeros((n_freqs, N, N), dtype=complex)
    z0 = np.full((n_freqs, N), 50.0)

    # Thru paths.
    for k, lane in enumerate(lane_s2p_list):
        p_pA, p_mB = 4 * k, 4 * k + 3
        s[:, p_mB, p_pA] = -2 * lane.s[:, 1, 0]
        s[:, p_pA, p_mB] = -2 * lane.s[:, 0, 1]

    # FEXT paths: from aggressor j input (+A=4j) to victim k output (-B=4k+3).
    for (j, k), fext_ntwk in fext_matrix.items():
        p_Aj = 4 * j        # aggressor j +A input
        p_Bk = 4 * k + 3   # victim k -B output
        s[:, p_Bk, p_Aj] = -2 * fext_ntwk.s[:, 1, 0]
        s[:, p_Aj, p_Bk] = -2 * fext_ntwk.s[:, 0, 1]

    return skrf.Network(f=freqs, s=s, z0=z0)


@pytest.fixture(scope="module")
def freqs() -> np.ndarray:
    return np.linspace(1e8, 10e9, 100)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImportFreqMultilane:
    def test_8port_lane0_s21_matches_reference(self, freqs, tmp_path):
        """Lane 0 extracted from an 8-port file should match the lane-0 reference S21."""
        lane0 = _make_tline_s2p(freqs, loss_db=2.0)
        lane1 = _make_tline_s2p(freqs, loss_db=5.0)  # distinct loss so lanes are distinguishable
        ntwk8 = _build_multilane_network([lane0, lane1])

        path = str(tmp_path / "test.s8p")
        ntwk8.write_touchstone(path)

        result = import_freq(path, lane=0)
        # sdd_21 of a matched differential pair with equal-amplitude, opposite-sign
        # conductors gives Sdd21 = S21 of either conductor.
        np.testing.assert_allclose(
            np.abs(result.s[:, 1, 0]),
            np.abs(lane0.s[:, 1, 0]),
            rtol=1e-6,
            err_msg="Lane 0 Sdd21 magnitude does not match reference",
        )

    def test_8port_lane1_s21_matches_reference(self, freqs, tmp_path):
        """Lane 1 extracted from an 8-port file should match the lane-1 reference S21."""
        lane0 = _make_tline_s2p(freqs, loss_db=2.0)
        lane1 = _make_tline_s2p(freqs, loss_db=5.0)
        ntwk8 = _build_multilane_network([lane0, lane1])

        path = str(tmp_path / "test.s8p")
        ntwk8.write_touchstone(path)

        result = import_freq(path, lane=1)
        np.testing.assert_allclose(
            np.abs(result.s[:, 1, 0]),
            np.abs(lane1.s[:, 1, 0]),
            rtol=1e-6,
            err_msg="Lane 1 Sdd21 magnitude does not match reference",
        )

    def test_12port_lane2_s21_matches_reference(self, freqs, tmp_path):
        """Lane 2 extracted from a 12-port file should match the lane-2 reference S21."""
        lanes = [_make_tline_s2p(freqs, loss_db=d) for d in (1.0, 3.0, 6.0)]
        ntwk12 = _build_multilane_network(lanes)

        path = str(tmp_path / "test.s12p")
        ntwk12.write_touchstone(path)

        result = import_freq(path, lane=2)
        np.testing.assert_allclose(
            np.abs(result.s[:, 1, 0]),
            np.abs(lanes[2].s[:, 1, 0]),
            rtol=1e-6,
            err_msg="Lane 2 Sdd21 magnitude does not match reference",
        )

    def test_8port_lane_out_of_range_raises(self, freqs, tmp_path):
        """Requesting lane 2 from an 8-port (2-lane) file must raise ValueError."""
        ntwk8 = _build_multilane_network([_make_tline_s2p(freqs)] * 2)
        path = str(tmp_path / "bad.s8p")
        ntwk8.write_touchstone(path)

        with pytest.raises(ValueError, match="lane=2 out of range"):
            import_freq(path, lane=2)

    def test_unsupported_port_count_raises(self, freqs, tmp_path):
        """A 6-port file must raise ValueError (not in supported set)."""
        s = np.zeros((len(freqs), 6, 6), dtype=complex)
        ntwk6 = skrf.Network(f=freqs, s=s, z0=50.0)
        path = str(tmp_path / "bad.s6p")
        ntwk6.write_touchstone(path)

        with pytest.raises(ValueError, match="1, 2, 4, 8, or 12"):
            import_freq(path)


class TestImportFext:
    """Verify FEXT extraction from multi-lane Touchstone files."""

    def test_8port_returns_one_fext_network(self, freqs, tmp_path):
        """An 8-port file with victim=lane 0 should yield exactly one FEXT network (lane 1)."""
        lanes = [_make_tline_s2p(freqs, loss_db=d) for d in (2.0, 5.0)]
        ntwk8 = _build_multilane_network_with_fext(lanes, {})
        path = str(tmp_path / "test.s8p")
        ntwk8.write_touchstone(path)

        result = import_fext(path, freqs, victim_lane=0)
        assert len(result) == 1

    def test_12port_returns_two_fext_networks(self, freqs, tmp_path):
        """A 12-port file with victim=lane 1 should yield two FEXT networks (lanes 0, 2)."""
        lanes = [_make_tline_s2p(freqs, loss_db=d) for d in (1.0, 3.0, 6.0)]
        ntwk12 = _build_multilane_network_with_fext(lanes, {})
        path = str(tmp_path / "test.s12p")
        ntwk12.write_touchstone(path)

        result = import_fext(path, freqs, victim_lane=1)
        assert len(result) == 2

    def test_fext_s21_matches_reference(self, freqs, tmp_path):
        """Extracted FEXT Sdd21 should match the reference FEXT 2-port network."""
        lane0 = _make_tline_s2p(freqs, loss_db=2.0)
        lane1 = _make_tline_s2p(freqs, loss_db=5.0)
        fext_ref = _make_tline_s2p(freqs, loss_db=20.0)  # heavy FEXT attenuation

        ntwk8 = _build_multilane_network_with_fext(
            [lane0, lane1],
            fext_matrix={(1, 0): fext_ref},  # aggressor=lane1, victim=lane0
        )
        path = str(tmp_path / "test.s8p")
        ntwk8.write_touchstone(path)

        result = import_fext(path, freqs, victim_lane=0)
        assert len(result) == 1
        np.testing.assert_allclose(
            np.abs(result[0].s[:, 1, 0]),
            np.abs(fext_ref.s[:, 1, 0]),
            rtol=1e-6,
            err_msg="FEXT Sdd21 magnitude does not match reference",
        )

    def test_non_multilane_file_returns_empty(self, freqs, tmp_path):
        """A 2-port or 4-port file should return an empty list (no FEXT)."""
        lane = _make_tline_s2p(freqs, loss_db=3.0)
        path = str(tmp_path / "test.s2p")
        lane.write_touchstone(path)

        assert import_fext(path, freqs, victim_lane=0) == []

    def test_victim_lane_out_of_range_raises(self, freqs, tmp_path):
        """victim_lane out of range for the file's lane count must raise ValueError."""
        lanes = [_make_tline_s2p(freqs)] * 2
        ntwk8 = _build_multilane_network_with_fext(lanes, {})
        path = str(tmp_path / "test.s8p")
        ntwk8.write_touchstone(path)

        with pytest.raises(ValueError, match="victim_lane=2 out of range"):
            import_fext(path, freqs, victim_lane=2)
