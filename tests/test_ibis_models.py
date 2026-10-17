"""PyBERT's handling of IBIS models: which `Model_type`s it offers, and which AMI files it picks for each side.

The IBIS files are small, generated ones (in the style of pyibisami's `tests/ibis/test_file.py`),
so each test controls the `Model_type`s and `Executable` lines it needs.
"""

import logging
import shutil
from pathlib import Path

import pytest

from pybert.mcp.server import inspect_ibis_model
from pybert.pybert import PyBERT
from pybert.utility.ibisami import PYBERT_RX_MODEL_TYPES, PYBERT_TX_MODEL_TYPES, PYIBISAMI_LOGGER
from pyibisami.ibis.file import RX_MODEL_TYPES, TX_MODEL_TYPES

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "models" / "ibisami"

HEADER = """[IBIS Ver]   5.1
[File Name]  {file_name}
[File Rev]   v0.1

[Component]    Comp
[Manufacturer] (n/a)

[Package]
R_pkg     0.10     0.00     0.50
L_pkg    10.00n    0.10n   50.00n
C_pkg     1.00p    0.01p    5.00p

[Pin]  signal_name        model_name
"""

RANGES = """
[Temperature_Range]     25.0      0.0    100.0
[Voltage_Range]         1.80     1.62     1.98
"""

DRIVER_TABLES = """
[Pulldown]
-1.80    -1.000e+01    -1.000e+01    -1.000e+01
0.00     0.000e+00     0.000e+00     0.000e+00
1.80     3.600e-02     4.000e-02     3.273e-02
3.60     1.000e+01     1.000e+01     1.000e+01

[Pullup]
-1.80    1.000e+01     1.000e+01     1.000e+01
0.00     -0.000e+00    -0.000e+00    -0.000e+00
1.80     -3.600e-02    -4.000e-02    -3.273e-02
3.60     -1.000e+01    -1.000e+01    -1.000e+01

[Ramp]
dV/dt_r    0.540/108.00p    0.512/511.58p    0.566/56.57p
dV/dt_f    0.540/108.00p    0.512/511.58p    0.566/56.57p
"""

CLAMP_TABLES = """
[GND Clamp]
-1.80    -1.000e+01    -1.000e+01    -1.000e+01
0.00     0.000e+00     0.000e+00     0.000e+00
1.80     3.600e-02     4.000e-02     3.273e-02

[POWER Clamp]
-1.80    1.000e+01     1.000e+01     1.000e+01
0.00     -0.000e+00    -0.000e+00    -0.000e+00
1.80     -3.600e-02    -4.000e-02    -3.273e-02
"""

# Tables each `Model_type` needs, for its `zout`/`zin`.
TABLES = {
    "Output": DRIVER_TABLES,
    "3-state": DRIVER_TABLES,
    "I/O": DRIVER_TABLES + CLAMP_TABLES,
    "Input": CLAMP_TABLES,
    "Open_drain": DRIVER_TABLES.replace(DRIVER_TABLES[DRIVER_TABLES.index("[Pullup]"):DRIVER_TABLES.index("[Ramp]")], ""),
}

# Both directions of an I/O model, for every platform the tests may run on.
IO_EXECUTABLES = """
[Algorithmic Model]
Executable_Tx linux_gcc4.1.2_64          tx.so    tx.ami
Executable_Tx Windows_VisualStudio_64    tx.dll   tx.ami
Executable_Rx linux_gcc4.1.2_64          rx.so    rx.ami
Executable_Rx Windows_VisualStudio_64    rx.dll   rx.ami
[End Algorithmic Model]
"""


def write_ibis(tmp_path: Path, models: dict[str, str], algo_model: str = "") -> str:
    """
    Write an IBIS file with one pin per model.

    Args:
        tmp_path: Directory to write it in.
        models: ``Model_type`` per model name.
        algo_model: ``[Algorithmic Model]`` block, added to every model.
    """
    ibis_file = tmp_path / "test.ibs"
    pins = "".join(f"{n}  sig_{name}  {name}\n" for n, name in enumerate(models, start=1))
    defs = "".join(
        f"\n[Model]   {name}\nModel_type   {mtype}\nC_comp  1.0p  0.9p  1.1p\nVmeas = 0.5\n"
        f"{algo_model}{RANGES}{TABLES[mtype]}"
        for name, mtype in models.items()
    )
    ibis_file.write_text(HEADER.format(file_name=ibis_file.name) + pins + defs + "\n[END]\n", encoding="utf-8")
    return str(ibis_file)


@pytest.fixture
def dut():
    return PyBERT(run_simulation=False, gui=False)


def test_pybert_model_types_are_spec_types():
    """PyBERT only narrows pyibisami's spec-derived driver/receiver sets; it never adds to them."""
    assert PYBERT_TX_MODEL_TYPES <= TX_MODEL_TYPES
    assert PYBERT_RX_MODEL_TYPES <= RX_MODEL_TYPES


def test_selectors_offer_only_supported_types(tmp_path, dut):
    """Each side's selector offers only the pins whose model PyBERT can simulate on that side."""
    ibis_file = write_ibis(tmp_path, {
        "drv": "Output", "tri": "3-state", "io": "I/O", "rcv": "Input", "od": "Open_drain",
    })
    dut.tx_ibis_file = ibis_file
    dut.rx_ibis_file = ibis_file
    assert dut.tx_ibis_valid and dut.rx_ibis_valid
    assert dut._tx_ibis.pins == ["1(sig_drv)", "2(sig_tri)", "3(sig_io)"]
    assert dut._rx_ibis.pins == ["3(sig_io)", "4(sig_rcv)"]


@pytest.mark.parametrize("side, mtype", [
    ("tx", "Input"),       # A receiver on the Tx side.
    ("tx", "Open_drain"),  # A driver PyBERT can't model (it needs an external pull-up).
    ("rx", "Output"),      # A driver on the Rx side.
])
def test_unsupported_model_type_is_rejected(tmp_path, dut, side, mtype):
    """A file with nothing PyBERT can simulate on a side leaves that side invalid, and says why."""
    ibis_file = write_ibis(tmp_path, {"only": mtype})
    dut.trait_set(**{f"{side}_ibis_file": ibis_file})  # Traits logs the handler's exception; it doesn't propagate.
    assert not getattr(dut, f"{side}_ibis_valid")
    assert f"No {side.capitalize()} model PyBERT can simulate" in dut.console_log
    assert f"Model_types found: ['{mtype}']" in dut.console_log


def test_io_model_on_rx_side_uses_zin(tmp_path, dut):
    """An I/O model is a valid receiver, terminated by its clamp impedance."""
    dut.rx_ibis_file = write_ibis(tmp_path, {"io": "I/O"})
    assert dut.rx_ibis_valid
    dut.rx_sel = "ibis"
    dut.calc_chnl_h()
    assert dut.RL == pytest.approx(dut._rx_ibis.model.zin * 2)


def test_io_model_gets_ami_files_per_direction(tmp_path, dut):
    """An I/O model's Tx side loads its `Executable_Tx` files, and its Rx side its `Executable_Rx` files."""
    for name in ("tx.ami", "rx.ami"):
        shutil.copy(EXAMPLE_DIR / "example_rx.ami", tmp_path / name)
    ibis_file = write_ibis(tmp_path, {"io": "I/O"}, algo_model=IO_EXECUTABLES)
    dut.tx_ibis_file = ibis_file
    dut.rx_ibis_file = ibis_file
    assert Path(dut.tx_ami_file) == tmp_path / "tx.ami"
    assert Path(dut.tx_dll_file).stem == "tx"
    assert Path(dut.rx_ami_file) == tmp_path / "rx.ami"
    assert Path(dut.rx_dll_file).stem == "rx"
    assert dut.tx_ami_valid and dut.rx_ami_valid


def test_relative_ibis_path_resolves_beside_ibis_file(dut, monkeypatch):
    """AMI files resolve against the IBIS file's directory, wherever PyBERT runs from."""
    monkeypatch.chdir(EXAMPLE_DIR.parent)
    dut.rx_ibis_file = "ibisami/example_rx.ibs"
    assert Path(dut.rx_ami_file) == EXAMPLE_DIR / "example_rx.ami"
    assert Path(dut.rx_dll_file).parent == EXAMPLE_DIR
    assert dut.rx_ami_valid and dut.rx_dll_valid


def test_mcp_reports_ami_files_per_direction(tmp_path):
    """`inspect_ibis_model` gives each direction's files for an I/O model, and `None` for values it lacks."""
    info = inspect_ibis_model(write_ibis(tmp_path, {"io": "I/O"}, algo_model=IO_EXECUTABLES), "io")
    assert Path(info["ami_file"]["tx"]) == tmp_path / "tx.ami"
    assert Path(info["ami_file"]["rx"]) == tmp_path / "rx.ami"
    assert Path(info["dll_file"]["rx"]).stem == "rx"

    info = inspect_ibis_model(write_ibis(tmp_path, {"rcv": "Input"}), "rcv")
    assert info["zout"] is None and info["zin"] is not None
    assert info["dll_file"] is None and info["ami_file"] is None


def test_mcp_reports_platform_ami_files():
    info = inspect_ibis_model(str(EXAMPLE_DIR / "example_rx.ibs"), "example_rx")
    assert Path(info["ami_file"]) == EXAMPLE_DIR / "example_rx.ami"
    assert Path(info["dll_file"]).parent == EXAMPLE_DIR


def test_debug_mode_shows_pyibisami_details(tmp_path, dut, caplog):
    """pyibisami's warnings are always shown; its parsing details only in debug mode."""
    ibis_file = write_ibis(tmp_path, {"drv": "Output"})
    (tmp_path / "again").mkdir()
    ibis_file_again = write_ibis(tmp_path / "again", {"drv": "Output"})
    caplog.set_level(logging.DEBUG)  # Capture whatever reaches the handlers; `debug` sets pyibisami's own level.

    dut.tx_ibis_file = ibis_file
    assert not any(r.name.startswith("pyibisami") and r.levelno == logging.DEBUG for r in caplog.records)

    caplog.clear()
    dut.debug = True
    dut.tx_ibis_file = ibis_file_again
    assert any(r.name == "pyibisami.ibis.file" and "Loading IBIS file" in r.message for r in caplog.records)

    dut.debug = False
    assert PYIBISAMI_LOGGER.getEffectiveLevel() == logging.WARNING


def test_pyibisami_handler_attached_once():
    """Every `PyBERT` instance shares one stderr handler (the MCP server makes one per call)."""
    PyBERT(run_simulation=False, gui=False)
    PyBERT(run_simulation=False, gui=False)
    handlers = [h for h in PYIBISAMI_LOGGER.handlers if isinstance(h, logging.StreamHandler)]
    assert len(handlers) == 1


TAP_AMI = r"""(tap_tx
    (Reserved_Parameters
         (Init_Returns_Impulse (Usage Info) (Type Boolean) (Value True))
         (GetWave_Exists (Usage Info) (Type Boolean) (Value False))
    )
    (Model_Specific
         (tx_tap
             (-1 (Usage In) (Type Tap) (Range 0.0 -0.5 0.5))
             (0  (Usage In) (Type Tap) (Range 1.0  0.5 1.0))
             (1  (Usage In) (Type Tap) (Range 0.0 -0.5 0.5))
         )
    )
)
"""


def test_ami_tap_tuners_keep_display_names_and_send_tap_numbers(tmp_path, dut):
    """Tap tuners show `pre1`/`post0`/`post1`, but the values PyBERT commits (as the optimizer does,
    through `tuner.branch_names`) reach the model under the `.ami` file's tap numbers."""
    ami_file = tmp_path / "tap_tx.ami"
    ami_file.write_text(TAP_AMI, encoding="utf-8")
    dut.tx_ami_file = str(ami_file)
    assert [t.name for t in dut.tx_ami_tap_tuners] == ["tx_tap_pre1", "tx_tap_post0", "tx_tap_post1"]
    pre1 = dut.tx_ami_tap_tuners[0]
    dut._tx_cfg.set_param_val(list(pre1.branch_names), -0.2)
    assert dut._tx_cfg.input_ami_params["tx_tap"] == {"-1": -0.2, "0": 1.0, "1": 0.0}
