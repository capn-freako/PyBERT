"""Unit test coverage for the `pybert.mcp` tool implementations."""

import asyncio
import math

import numpy as np
import pytest
import skrf

from pybert.mcp.server import (
    get_config,
    get_default_config,
    inspect_ibis_model,
    inspect_results_file,
    list_ami_params,
    list_ibis_models,
    optimize_eq,
    run_simulation,
    set_config,
)

IBIS_FILE = "models/ibisami/example_rx.ibs"


@pytest.fixture(scope="module")
def channel_segment(tmp_path_factory) -> str:
    """A synthetic, matched 0.1 m transmission-line segment, as real loadable
    S-parameter data. (The repo-root `dummy.s4p`/`dummy1.S4P` files are *empty* --
    fine for file-path-trait plumbing tests, useless for actually simulating a
    channel -- so this generates a real one, the same way
    `tests/test_composite_channel.py::two_segment_files` does.)
    """
    fs = np.linspace(1e9, 20e9, 256)
    alpha = 2.0 * np.sqrt(fs / 1e9) * (np.log(10) / 20)  # ~2 dB/m @ 1 GHz, skin-effect scaling
    beta = 2 * np.pi * fs / 2e8  # v = 2e8 m/s
    s21 = np.exp(-(alpha + 1j * beta) * 0.10)  # 0.10 m
    s = np.zeros((len(fs), 2, 2), dtype=complex)
    s[:, 0, 1] = s21
    s[:, 1, 0] = s21
    ntwk = skrf.Network(f=fs, s=s, z0=50.0)
    path = tmp_path_factory.mktemp("mcp_channel") / "segment.s2p"
    ntwk.write_touchstone(str(path))
    return str(path)


def _fast_config() -> dict:
    """A default config, shrunk down so `run_simulation` is fast in tests."""
    cfg = get_default_config()
    cfg["nbits"] = 1000
    cfg["eye_bits"] = 500
    return cfg


def _with_channel(cfg: dict, *files: str) -> dict:
    """Point `cfg` at one or more channel touchstone files, cascaded via `inter_sel =
    "multiple"` when more than one is given (the multi-segment channel capability --
    see `tests/test_composite_channel.py`). `inter_sel` defaults to "native" (the
    analytic Howard Johnson UTP model), which silently ignores a bare `ch_files` list
    if left unset -- so this always sets it explicitly, even for a single file.
    """
    cfg["inter_sel"] = "multiple"
    cfg["ch_files"] = list(files)
    cfg["ch_file"] = ""
    return cfg


def test_run_simulation_returns_metrics():
    """`run_simulation` reports a finite `total_perf` and the expected metric keys."""
    result = run_simulation(_fast_config())
    assert result["status"]
    assert math.isfinite(result["total_perf"])
    assert math.isfinite(result["jitter_perf"])


def test_run_simulation_multi_segment_channel(channel_segment):
    """`run_simulation` accepts a multi-segment (cascaded) channel -- `ch_files` with
    more than one entry and `inter_sel = "multiple"` -- not just a single file."""
    cfg = _with_channel(_fast_config(), channel_segment, channel_segment)
    result = run_simulation(cfg)
    assert result["status"]
    assert math.isfinite(result["total_perf"])
    assert result["chnl_dly"] > 0


def test_run_simulation_saves_results(tmp_path):
    """`results_file`, when given, is written and can be inspected afterward."""
    results_file = tmp_path / "results.pybert_data"
    run_simulation(_fast_config(), results_file=str(results_file))
    assert results_file.exists()

    summary = inspect_results_file(str(results_file))
    assert summary["chnl_h"]["shape"][0] > 0
    assert math.isfinite(summary["chnl_h"]["mean"])


def test_get_set_config_round_trip(tmp_path):
    """Overrides applied by `set_config` are visible via `get_config`."""
    config_file = tmp_path / "config.yaml"
    saved = set_config({"bit_rate": 25, "nbits": 2000}, str(config_file))
    assert saved["config"]["bit_rate"] == 25

    reloaded = get_config(str(config_file))
    assert reloaded["bit_rate"] == 25
    assert reloaded["nbits"] == 2000


def test_set_config_rejects_unknown_key(tmp_path):
    """An override key that isn't a real PyBERT config attribute raises `ValueError`."""
    with pytest.raises(ValueError, match="not_a_real_key"):
        set_config({"not_a_real_key": 1}, str(tmp_path / "config.yaml"))


def test_run_simulation_rejects_unknown_key():
    """A typo in `config` (e.g. from an LLM caller) fails loudly rather than being ignored."""
    config = _fast_config()
    config["bit_rat"] = 25  # typo of "bit_rate"
    with pytest.raises(ValueError, match="bit_rat"):
        run_simulation(config)


def test_optimize_eq_updates_config(channel_segment):
    """`optimize_eq` returns Tx-FFE/CTLE values already baked into a `config` that
    `run_simulation` can consume directly."""
    cfg = _with_channel(_fast_config(), channel_segment)

    result = optimize_eq(cfg)
    assert math.isfinite(result["fom"])
    assert len(result["tx_weights"]) == len(cfg["tx_taps"])
    assert result["config"]["peak_mag"] == result["rx_peaking"]

    sim = run_simulation(result["config"])
    assert sim["status"]


def test_optimize_eq_rejects_unknown_key(channel_segment):
    """Same guardrail as `run_simulation`: a typo in `config` fails loudly."""
    cfg = _with_channel(_fast_config(), channel_segment)
    cfg["bit_rat"] = 25  # typo of "bit_rate"
    with pytest.raises(ValueError, match="bit_rat"):
        optimize_eq(cfg)


def test_optimize_eq_accepts_max_trials(channel_segment):
    """`max_trials` is accepted (and, on the native-EQ path exercised here, simply
    irrelevant -- native EQ is an exact solve, not a trial search)."""
    cfg = _with_channel(_fast_config(), channel_segment)
    result = optimize_eq(cfg, max_trials=3)
    assert math.isfinite(result["fom"])


def test_optimize_eq_rejects_bad_max_trials(channel_segment):
    """`max_trials` must be a positive integer."""
    cfg = _with_channel(_fast_config(), channel_segment)
    with pytest.raises(ValueError, match="max_trials"):
        optimize_eq(cfg, max_trials=0)


def _rx_ami_config(channel_segment: str) -> dict:
    """A fast config with the repo-bundled example Rx IBIS-AMI model selected.

    Note: Rx-AMI activation goes through `rx_eq_sel = "IBIS-AMI"`, not `rx_use_ami`
    directly -- `_rx_ibis_file_changed` resets `rx_use_ami` to False as a side effect
    of (re-)parsing the IBIS file, and it's `_rx_eq_sel_changed` that turns it back on
    once the file's parsed valid (see `src/pybert/pybert.py`).
    """
    cfg = _with_channel(_fast_config(), channel_segment)
    cfg["rx_ibis_file"] = IBIS_FILE
    cfg["rx_sel"] = "ibis"
    cfg["rx_eq_sel"] = "IBIS-AMI"
    return cfg


def test_list_ami_params_reports_rx_model_specific_params(channel_segment):
    """`list_ami_params` surfaces the Rx AMI model's declared parameter names, all
    disabled by default (opt-in only), and nothing on the (native) Tx side."""
    info = list_ami_params(_rx_ami_config(channel_segment))
    names = {p["name"] for p in info["rx"]}
    assert {"ctle_mag", "ctle_mode", "dfe_gain"} <= names
    assert all(not p["enabled"] for p in info["rx"])
    assert info["tx"] == []


def test_optimize_eq_rx_ami_params_selects_search_space(channel_segment):
    """`rx_ami_params` enables exactly the named AMI parameters for the search; a real
    optimum is found for the known-good `ctle_mode`/`ctle_mag` combo (mirrors
    `tests/test_optimizer_ami.py::TestRxAmiOptimizer::test_mode_selector_gates_sibling_range_param`)."""
    cfg = _rx_ami_config(channel_segment)
    result = optimize_eq(cfg, max_trials=15, rx_ami_params=["ctle_mode", "ctle_mag"])

    names = [p["name"] for p in list_ami_params(cfg)["rx"]]
    ctle_mode_val = result["rx_ami_vals"][names.index("ctle_mode")]
    ctle_mag_val = result["rx_ami_vals"][names.index("ctle_mag")]
    assert ctle_mode_val == 1.0  # "Manual" -- CTLE actually engaged.
    assert ctle_mag_val > 0.0  # A genuine, non-trivial optimum was found.
    assert result["config"]["rx_eq_sel"] == "IBIS-AMI"


def test_optimize_eq_rejects_unknown_ami_param_name(channel_segment):
    """A typo in `rx_ami_params` fails loudly, listing the real available names."""
    cfg = _rx_ami_config(channel_segment)
    with pytest.raises(ValueError, match="not_a_real_param"):
        optimize_eq(cfg, rx_ami_params=["not_a_real_param"])


def test_list_ibis_models():
    """`list_ibis_models` reports the components/models parsed out of an IBIS file."""
    info = list_ibis_models(IBIS_FILE)
    assert "example_rx" in info["models"]
    assert info["parsing_errors"] == "Success!"


def test_inspect_ibis_model():
    """`inspect_ibis_model` reports AMI file references for an AMI-enabled model."""
    info = inspect_ibis_model(IBIS_FILE, "example_rx")
    assert info["is_ami"] is True
    assert info["ami_files"]["64-bit"]["lin"]


def test_build_server_registers_all_tools():
    """The MCP server exposes every tool function."""
    pytest.importorskip("mcp")
    from pybert.mcp.server import build_server

    server = build_server()
    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}
    assert {
        "get_default_config",
        "get_config",
        "set_config",
        "list_ami_params",
        "optimize_eq",
        "run_simulation",
        "inspect_results_file",
        "list_ibis_models",
        "inspect_ibis_model",
    } <= tool_names
