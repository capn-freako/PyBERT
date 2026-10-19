"""Unit tests for VITA 68.x-convention multi-lane Touchstone import.

A VITA 68.x reference-model segment (Module or Backplane) bundles *both* directions
of traffic for a connector position into one file, so each lane occupies an 8-port
block -- not the plain 4-port "(+A,+B,-A,-B)" block ``import_freq`` otherwise assumes
(see its module-level notes in ``src/pybert/utility/sparam.py``). Within a lane's
8-port block (0-indexed offsets), ``vita_end="tx"`` selects offsets (4,6,5,7) and
``vita_end="rx"`` selects offsets (1,3,0,2); both were derived from real VITA 68.3
reference-model Touchstone header comments (Module: P/C x RX/TX; Backplane: L/R x
RX/TX) and verified empirically against the construction technique below.

Construction technique: same "-2T at a single element" trick as
``tests/test_multilane_sparam.py`` -- after ``import_freq`` extracts a lane's 4
relevant ports in local order (0,1,2,3) and remaps them to sdd_21's expected
(+A,+B,-A,-B) input order, setting global ``S[local3, local0] = -2T`` (plus its
reciprocal) makes the extracted ``Sdd21`` equal ``T`` exactly.
"""

import numpy as np
import pytest
import skrf

from pybert.utility.sparam import import_channel, import_freq

_VITA_PORTS_PER_LANE = 8
# Local (0,1,2,3) -> global-offset-within-lane-block, per `vita_end`.
_TX_LOCAL_TO_OFFSET = (4, 6, 5, 7)
_RX_LOCAL_TO_OFFSET = (1, 3, 0, 2)


def _build_vita_network(n_lanes: int, tx_values: dict, rx_values: dict,
                         n_freqs: int = 8) -> skrf.Network:
    """Build an `8 * n_lanes`-port synthetic VITA-style network.

    ``tx_values``/``rx_values`` map lane index -> desired Sdd21 (a complex scalar,
    constant across frequency) for that lane's tx-direction / rx-direction path.
    Un-mentioned lanes/directions are left at zero (fully isolated), so a test can
    verify that extracting lane k picks up *only* lane k's assigned value.
    """
    fs = np.linspace(1e9, 10e9, n_freqs)
    n_ports = _VITA_PORTS_PER_LANE * n_lanes
    s = np.zeros((n_freqs, n_ports, n_ports), dtype=complex)

    def _set(lane: int, local_to_offset: tuple, value: complex) -> None:
        base = _VITA_PORTS_PER_LANE * lane
        g0, g3 = base + local_to_offset[0], base + local_to_offset[3]
        s[:, g3, g0] = -2 * value
        s[:, g0, g3] = -2 * value

    for lane, val in tx_values.items():
        _set(lane, _TX_LOCAL_TO_OFFSET, val)
    for lane, val in rx_values.items():
        _set(lane, _RX_LOCAL_TO_OFFSET, val)

    return skrf.Network(f=fs, s=s, z0=50.0)


class TestVitaLaneExtraction:
    "Correct lane/direction is extracted, and other lanes/directions stay isolated."

    def test_single_lane_tx(self, tmp_path):
        T = 0.7 + 0.1j
        ntwk = _build_vita_network(1, tx_values={0: T}, rx_values={})
        f = tmp_path / "seg.s8p"
        ntwk.write_touchstone(str(f))

        out = import_freq(str(f), lane=0, vita_end="tx")
        assert out.s[:, 1, 0] == pytest.approx(T)

    def test_single_lane_rx(self, tmp_path):
        T = 0.4 - 0.2j
        ntwk = _build_vita_network(1, tx_values={}, rx_values={0: T})
        f = tmp_path / "seg.s8p"
        ntwk.write_touchstone(str(f))

        out = import_freq(str(f), lane=0, vita_end="rx")
        assert out.s[:, 1, 0] == pytest.approx(T)

    def test_tx_and_rx_are_independent_within_one_lane(self, tmp_path):
        """A lane's own tx-path value must not leak into its rx-path extraction,
        and vice versa (they occupy disjoint ports within the 8-port block)."""
        tx_val, rx_val = 0.8 + 0j, 0.1 + 0j
        ntwk = _build_vita_network(1, tx_values={0: tx_val}, rx_values={0: rx_val})
        f = tmp_path / "seg.s8p"
        ntwk.write_touchstone(str(f))

        tx_out = import_freq(str(f), lane=0, vita_end="tx")
        rx_out = import_freq(str(f), lane=0, vita_end="rx")
        assert tx_out.s[:, 1, 0] == pytest.approx(tx_val)
        assert rx_out.s[:, 1, 0] == pytest.approx(rx_val)

    def test_multi_lane_isolation(self, tmp_path):
        "Extracting lane 1 of a 4-lane (32-port) file picks up only lane 1's value."
        values = {0: 0.9 + 0j, 1: 0.5 + 0j, 2: 0.2 + 0j, 3: 0.05 + 0j}
        ntwk = _build_vita_network(4, tx_values=values, rx_values={})
        f = tmp_path / "seg.s32p"
        ntwk.write_touchstone(str(f))

        for lane, expected in values.items():
            out = import_freq(str(f), lane=lane, vita_end="tx")
            assert out.s[:, 1, 0] == pytest.approx(expected), f"lane {lane}"

    def test_import_channel_passthrough(self, tmp_path):
        "`import_channel` (the higher-level entry point) forwards `vita_end` correctly."
        T = 0.6 + 0j
        ntwk = _build_vita_network(1, tx_values={0: T}, rx_values={})
        f = tmp_path / "seg.s8p"
        ntwk.write_touchstone(str(f))

        fs = np.linspace(1e9, 10e9, 8)
        out = import_channel(str(f), sample_per=1e-11, fs=fs, vita_end="tx", lane=0)
        assert abs(out.s[0, 1, 0]) == pytest.approx(abs(T), rel=1e-2)


class TestVitaValidation:
    "Error handling for malformed inputs."

    def test_rejects_non_multiple_of_8(self, tmp_path):
        ntwk = _build_vita_network(1, tx_values={0: 0.5 + 0j}, rx_values={})
        # Drop to a 4-port file (not a multiple of 8) while keeping vita_end set.
        s4 = ntwk.s[:, :4, :4]
        bad = skrf.Network(f=ntwk.f, s=s4, z0=50.0)
        f = tmp_path / "bad.s4p"
        bad.write_touchstone(str(f))
        with pytest.raises(ValueError, match="multiple of 8"):
            import_freq(str(f), lane=0, vita_end="tx")

    def test_rejects_unknown_end(self, tmp_path):
        ntwk = _build_vita_network(1, tx_values={0: 0.5 + 0j}, rx_values={})
        f = tmp_path / "seg.s8p"
        ntwk.write_touchstone(str(f))
        with pytest.raises(ValueError, match="'tx' or 'rx'"):
            import_freq(str(f), lane=0, vita_end="bogus")

    def test_rejects_lane_out_of_range(self, tmp_path):
        ntwk = _build_vita_network(2, tx_values={0: 0.5 + 0j}, rx_values={})
        f = tmp_path / "seg.s16p"
        ntwk.write_touchstone(str(f))
        with pytest.raises(ValueError, match="out of range"):
            import_freq(str(f), lane=5, vita_end="tx")


class TestVitaBackwardCompatibility:
    "`vita_end=None` (the default) leaves existing behavior completely unchanged."

    def test_plain_4port_unaffected(self, tmp_path):
        fs = np.linspace(1e9, 10e9, 8)
        nf = len(fs)
        s = np.zeros((nf, 4, 4), dtype=complex)
        g = 0.9 * np.ones(nf)
        s[:, 1, 0] = g; s[:, 0, 1] = g
        s[:, 3, 2] = g; s[:, 2, 3] = g
        ntwk = skrf.Network(f=fs, s=s, z0=50.0)
        f = tmp_path / "plain.s4p"
        ntwk.write_touchstone(str(f))

        out = import_freq(str(f))  # vita_end left at its default (None)
        assert abs(out.s[0, 1, 0]) == pytest.approx(0.9, rel=1e-2)
