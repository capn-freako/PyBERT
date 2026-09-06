"""Unit test coverage for the `pybert.mcp` tool implementations."""

import asyncio
import math

import pytest

from pybert.mcp.server import (
    get_config,
    get_default_config,
    inspect_ibis_model,
    inspect_results_file,
    list_ibis_models,
    run_simulation,
    set_config,
)

IBIS_FILE = "models/ibisami/example_rx.ibs"


def _fast_config() -> dict:
    """A default config, shrunk down so `run_simulation` is fast in tests."""
    cfg = get_default_config()
    cfg["nbits"] = 1000
    cfg["eye_bits"] = 500
    return cfg


def test_run_simulation_returns_metrics():
    """`run_simulation` reports a finite `total_perf` and the expected metric keys."""
    result = run_simulation(_fast_config())
    assert result["status"]
    assert math.isfinite(result["total_perf"])
    assert math.isfinite(result["jitter_perf"])


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
        "run_simulation",
        "inspect_results_file",
        "list_ibis_models",
        "inspect_ibis_model",
    } <= tool_names
