"""Tests of the P802.3dj Annex 178A Tx/Rx device models in the channel (`pkg_enable`)."""
import numpy as np
import pytest

import pybert.pybert as pybert_module
from pybert.pybert import PyBERT
from pychopmarg.devmodel import DeviceModel, DeviceTermination


def _native(**traits) -> PyBERT:
    dut = PyBERT(run_simulation=False, gui=False)
    for k, v in traits.items():
        setattr(dut, k, v)
    return dut


def test_zero_size_device_model_reproduces_calc_g(monkeypatch):
    """With device models reduced to a bare, matched termination (no C or L, Rd = R0 = rs/2 = rin/2), the packaged
    channel's transfer function equals `calc_G()`'s (same open-circuit-voltage reference) for the same channel."""
    from pybert.utility.channel import calc_G
    bare = DeviceModel(termination=DeviceTermination(Cd=(0.0,), Ls=(0.0,), Cb=0.0), Rd=50.0)
    monkeypatch.setattr(pybert_module, "device_model", lambda *args, **kwargs: bare)
    monkeypatch.setattr(pybert_module, "DJ_R0", 50.0)
    dut = _native(rs=100.0, rin=100.0, pkg_enable=True)
    dut.calc_chnl_h()
    ch = dut.ch_s2p_pre
    ref = calc_G(ch.s21.s.flatten(), 100.0, 0.0, ch.s21.z0.flatten() * np.ones(len(dut.w)), 100.0, 0.0, dut.w)
    np.testing.assert_allclose(dut.chnl_H[1:], ref[1:], rtol=1e-5)  # (calc_G() swaps a 0 F Cp for 1e-18 F.)


def test_kr_package_adds_loss_and_delay():
    """KR class A Test 1 packages on both ends add loss at Nyquist and ~2 x 35 mm x 6.141 ps/mm of delay."""
    base = _native(bit_rate=212.5, mod_type="PAM-4", f_max=120.0, rs=92.5, rin=92.5, cout=0.001, cin=0.001)
    base.calc_chnl_h()
    pkg = _native(bit_rate=212.5, mod_type="PAM-4", f_max=120.0, pkg_enable=True, pkg_group="KR",
                  tx_pkg_case="A1", rx_pkg_case="A1")
    pkg.calc_chnl_h()
    f = pkg.f
    nyq = np.argmin(abs(f - 53.125e9))
    assert abs(pkg.chnl_H[nyq]) < abs(base.chnl_H[nyq])
    dly = pkg.chnl_dly - base.chnl_dly
    assert 0.3e-9 < dly < 0.6e-9, dly


def test_pkg_case_validation():
    dut = _native(pkg_enable=True, pkg_group="KR", tx_pkg_case="HL", rx_pkg_case="A1")
    with pytest.raises(ValueError):
        dut.calc_chnl_h()


def test_pkg_traits_round_trip(tmp_path):
    dut = _native(pkg_enable=True, pkg_group="CR", tx_pkg_case="HN", rx_pkg_case="HH", pkg_include_host=False)
    cfg = tmp_path / "pkg.yaml"
    dut.save_configuration(cfg)
    new = PyBERT(run_simulation=False, gui=False)
    new.load_configuration(cfg)
    assert (new.pkg_enable, new.pkg_group, new.tx_pkg_case, new.rx_pkg_case, new.pkg_include_host) == \
        (True, "CR", "HN", "HH", False)


@pytest.mark.parametrize("pdut", [{"pkg_enable": True, "pkg_group": "KR", "tx_pkg_case": "A2",
                                   "rx_pkg_case": "A2"}], indirect=True)
def test_simulation_with_packages(pdut):
    "The default (native channel) simulation still runs error-free with KR packages on both ends."
    assert pdut.status == "Ready."
    assert pdut.n_errs_dfe == 0
