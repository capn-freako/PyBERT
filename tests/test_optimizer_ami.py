"""Regression tests for EQ optimization against IBIS-AMI Tx/Rx models.

Exercises the `tx_use_ami`/`rx_use_ami` paths added to `coopt()`
(`src/pybert/threads/optimization.py`), using real compiled DLL/SO models --
not mocks -- so these tests catch problems (like the initial `identity_h`
length bug caught during development) that a purely-native-EQ test can't.

- Rx side: the real, repo-bundled `models/ibisami/example_rx.*`, loaded via
  the normal `rx_ibis_file`/`rx_sel`/`rx_use_ami` property-change chain, so
  the AMI-file-changed handlers' `rx_ami_tap_tuners` population is exercised
  too.
- Tx side: PyAMI's bundled `example_tx_*` example model (no standalone Tx
  IBIS file ships in this repo), wired up directly onto `_tx_cfg`.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pybert.pybert import PyBERT
from pybert.threads.optimization import coopt
from pyibisami.ami.parser import AMIParamConfigurator

RX_IBIS_FILE = "models/ibisami/example_rx.ibs"

if sys.platform == "win32":
    _tx_so_name = "example_tx_x86_amd64.dll"
elif sys.platform.startswith("linux"):
    _tx_so_name = "example_tx_x86_amd64.so"
else:  # darwin
    _tx_so_name = "example_tx_x86_amd64_osx.so"
TX_SO_PATH = Path(__file__).resolve().parents[1] / "PyAMI" / "tests" / "examples" / _tx_so_name

TX_AMI_CONTENT = r"""(example_tx
    (Reserved_Parameters
         (Init_Returns_Impulse (Usage Info) (Type Boolean) (Value True) (Description "x"))
         (GetWave_Exists (Usage Info) (Type Boolean) (Value True) (Description "x"))
    )
    (Model_Specific
         (tx_tap_np1 (Usage In) (Type Integer) (Range 0 0 10) (Description "x"))
         (tx_tap_nm1 (Usage In) (Type Integer) (Range 0 0 10) (Description "x"))
    )
)
"""

needs_tx_so = pytest.mark.skipif(not TX_SO_PATH.exists(), reason=f"Example Tx DLL/SO not found: {TX_SO_PATH}")


def _mk_rx_ami_dut() -> PyBERT:
    dut = PyBERT(run_simulation=False, gui=False)
    dut.rx_ibis_file = RX_IBIS_FILE
    dut.rx_sel = "ibis"
    dut.rx_use_ami = True
    return dut


class TestRxAmiTunerWiring:
    "The Rx-AMI file-changed handler should populate `rx_ami_tap_tuners` automatically."

    def test_tuners_populated_from_ami_file(self):
        dut = _mk_rx_ami_dut()
        names = {t.name for t in dut.rx_ami_tap_tuners}
        # Range-format parameters...
        assert {"ctle_mag", "ctle_bandwidth", "ctle_dcgain", "dfe_vout", "dfe_gain"} <= names
        # ...Boolean parameters...
        assert "debug_dbg_enable" in names
        # ...and contiguous Integer/List "mode selector" parameters (these gate whether
        # their sibling Range parameters have *any* effect -- e.g. ctle_mag is a no-op
        # unless ctle_mode is also swept/enabled) all show up.
        assert "ctle_mode" in names
        assert "dfe_mode" in names
        # Opt-in only: nothing enabled by default.
        assert all(not t.enabled for t in dut.rx_ami_tap_tuners)


class TestRxAmiOptimizer:
    "Rx is a real IBIS-AMI model (real AMI_Init() calls); Tx stays native."

    def test_optimize_with_rx_ami_enabled(self):
        dut = _mk_rx_ami_dut()
        for tuner in dut.rx_ami_tap_tuners:
            if tuner.name in ("ctle_mag", "dfe_gain"):
                tuner.enabled = True
                tuner.step = (tuner.max_val - tuner.min_val) / 2 or 1.0

        (tx_weights_best, _peak_mag_best, rx_weights_best, fom_max,
         valid, tx_ami_best, rx_ami_best) = coopt(dut)

        assert valid
        assert fom_max > -1000.0
        assert tx_weights_best  # Native Tx untouched -> non-empty tap-weight list.
        assert all(w == 0.0 for w in rx_weights_best)  # No native Rx FFE when Rx is AMI.
        assert len(rx_ami_best) == len(dut.rx_ami_tap_tuners)

        # Winning values must already be committed into the live `_rx_cfg` (this *is* the
        # "Use EQ" step, for AMI parameters) -- re-derive tuners from `_rx_cfg`'s current
        # state (handles Range/Boolean/List-format params uniformly) and compare.
        committed_tuners = dut._rx_cfg.mk_tap_tuners()
        for tuner, val, committed in zip(dut.rx_ami_tap_tuners, rx_ami_best, committed_tuners):
            assert val == committed.value
            if not tuner.enabled:
                assert val == tuner.value  # Disabled params hold at their configured value.

    def test_mode_selector_gates_sibling_range_param(self):
        """Regression: a real-world 'mode selector' pattern. `ctle_mag` has *no* effect on
        the model unless `ctle_mode` is also set to 'Manual' -- enabling only `ctle_mag`
        used to silently optimize to a meaningless result (always the sweep's first/seed
        value, since no candidate could ever out-score another)."""
        dut = _mk_rx_ami_dut()
        for tuner in dut.rx_ami_tap_tuners:
            if tuner.name in ("ctle_mode", "ctle_mag"):
                tuner.enabled = True

        (*_rest, valid, _tx_ami_best, rx_ami_best) = coopt(dut)

        assert valid
        names = [t.name for t in dut.rx_ami_tap_tuners]
        ctle_mode_val = rx_ami_best[names.index("ctle_mode")]
        ctle_mag_val = rx_ami_best[names.index("ctle_mag")]
        assert ctle_mode_val == 1.0  # "Manual" -- CTLE actually engaged.
        assert ctle_mag_val > 0.0  # A genuine, non-trivial optimum was found.
        assert dut._rx_cfg.input_ami_params["ctle_mode"] == 1
        assert dut._rx_cfg.input_ami_params["ctle_mag"] == ctle_mag_val


@needs_tx_so
class TestTxAmiOptimizer:
    "Tx is a real IBIS-AMI model (real AMI_Init() calls); Rx stays native (MMSE)."

    def test_optimize_with_tx_ami_enabled(self):
        dut = PyBERT(run_simulation=False, gui=False)
        pcfg = AMIParamConfigurator(TX_AMI_CONTENT)
        dut._tx_cfg = pcfg
        dut.tx_dll_file = str(TX_SO_PATH)
        dut.tx_ami_tap_tuners = pcfg.mk_tap_tuners()
        dut.tx_use_ami = True
        for tuner in dut.tx_ami_tap_tuners:
            tuner.enabled = True
            tuner.step = 5

        (tx_weights_best, _peak_mag_best, _rx_weights_best, fom_max,
         valid, tx_ami_best, rx_ami_best) = coopt(dut)

        assert valid
        assert fom_max > -1000.0
        assert not tx_weights_best  # Native Tx not swept -> empty.
        assert not len(rx_ami_best)  # Rx is native.
        assert len(tx_ami_best) == len(dut.tx_ami_tap_tuners)

        for tuner, val in zip(dut.tx_ami_tap_tuners, tx_ami_best):
            committed = pcfg.fetch_param_val(list(tuner.branch_names))
            assert val == committed

    def test_getwave_only_model_is_skipped_gracefully(self):
        "A model with Init_Returns_Impulse=False can't be swept; must fall back cleanly."
        getwave_only_content = TX_AMI_CONTENT.replace(
            "(Init_Returns_Impulse (Usage Info) (Type Boolean) (Value True)",
            "(Init_Returns_Impulse (Usage Info) (Type Boolean) (Value False)")
        dut = PyBERT(run_simulation=False, gui=False)
        pcfg = AMIParamConfigurator(getwave_only_content)
        dut._tx_cfg = pcfg
        dut.tx_dll_file = str(TX_SO_PATH)
        dut.tx_ami_tap_tuners = pcfg.mk_tap_tuners()
        dut.tx_use_ami = True
        for tuner in dut.tx_ami_tap_tuners:
            tuner.enabled = True

        logs = []
        dut.log = lambda msg, **kw: logs.append(msg)

        *_rest, valid, tx_ami_best, _rx_ami_best = coopt(dut)

        assert valid
        assert any("GetWave" in msg for msg in logs)
        # Falls back to a single "as configured" pass -- values pass through unchanged.
        assert list(tx_ami_best) == [t.value for t in dut.tx_ami_tap_tuners]

    def test_empty_search_space_skips_optuna(self):
        "Nothing enabled anywhere: one fixed evaluation, no Optuna study created."
        dut = PyBERT(run_simulation=False, gui=False)
        pcfg = AMIParamConfigurator(TX_AMI_CONTENT)
        dut._tx_cfg = pcfg
        dut.tx_dll_file = str(TX_SO_PATH)
        dut.tx_ami_tap_tuners = pcfg.mk_tap_tuners()  # All disabled by default.
        dut.tx_use_ami = True
        dut.ctle_enable_tune = False  # Zero out the native Rx-side dimension too.

        with patch("optuna.create_study") as mock_create_study:
            (tx_weights_best, _peak_mag_best, _rx_weights_best, fom_max,
             valid, tx_ami_best, rx_ami_best) = coopt(dut)

        mock_create_study.assert_not_called()
        assert valid
        assert fom_max > -1000.0
        assert not tx_weights_best
        assert not len(rx_ami_best)
        # The single fixed evaluation used each tuner's current (unswept) value.
        assert list(tx_ami_best) == [t.value for t in dut.tx_ami_tap_tuners]
