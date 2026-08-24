"""MCP tool implementations for PyBERT, and the MCP server that exposes them.

Every tool constructs its own, fresh ``PyBERT(gui=False)`` instance. Nothing is cached or
shared across calls: ``PyBERT`` is a large, mutable ``HasTraits`` object, and simulations can
take a while, so cross-call state would be a correctness hazard for no benefit.
"""
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pybert import __version__ as VERSION
from pybert.configuration import PyBertCfg
from pybert.pybert import PyBERT
from pyibisami.ibis.file import IBISModel

RESULT_TRAIT_NAMES = [
    "status",
    "total_perf",
    "jitter_perf",
    "n_errs_dfe",
    "n_errs_viterbi",
    "chnl_dly",
    "isi_dfe",
    "dcd_dfe",
    "pj_dfe",
    "rj_dfe",
    "perf_info",
    "jitter_info",
]


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/tuple values into plain JSON-serializable types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _config_dict(pybert: PyBERT) -> dict:
    cfg = PyBertCfg(pybert, time.asctime(), VERSION)
    return _jsonify(vars(cfg))


def _apply_config(pybert: PyBERT, config: dict) -> None:
    """Apply a config dict (full or partial, as produced by `_config_dict`) onto `pybert`.

    Round-trips through a temp YAML file and `PyBERT.load_configuration()` so that
    `PyBertCfg.load_from_file`'s existing special-casing (tap-tuple lists, legacy field
    renames) is reused rather than reimplemented here.

    Raises:
        ValueError: If `config` contains a key that isn't a real PyBERT config attribute.
    """
    valid_keys = set(vars(PyBertCfg(pybert, time.asctime(), VERSION)))
    unknown = set(config) - valid_keys
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {sorted(unknown)}")

    cfg = PyBertCfg.__new__(PyBertCfg)  # bypass __init__; it requires a live PyBERT to copy from
    cfg.__dict__.update(config)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp, indent=4, sort_keys=False)
        tmp_path = Path(tmp.name)
    try:
        pybert.load_configuration(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def get_default_config() -> dict:
    """Return PyBERT's default configuration, as a JSON-serializable dict."""
    return _config_dict(PyBERT(run_simulation=False, gui=False))


def get_config(config_file: str) -> dict:
    """Load a saved configuration file and return its contents as a dict."""
    pybert = PyBERT(run_simulation=False, gui=False)
    pybert.load_configuration(Path(config_file))
    return _config_dict(pybert)


def set_config(overrides: dict, out_file: str, base_config_file: str | None = None) -> dict:
    """Apply ``overrides`` on top of a base configuration and save the result.

    Args:
        overrides: Trait name/value pairs to change.
        out_file: Where to save the resulting configuration (``.yaml``).
        base_config_file: An existing config file to start from; PyBERT's defaults if omitted.
    """
    pybert = PyBERT(run_simulation=False, gui=False)
    if base_config_file:
        pybert.load_configuration(Path(base_config_file))

    _apply_config(pybert, overrides)
    pybert.save_configuration(Path(out_file))
    return {"saved": out_file, "config": _config_dict(pybert)}


def run_simulation(config: dict, results_file: str | None = None) -> dict:
    """Run a headless PyBERT simulation and return its summary performance metrics.

    Args:
        config: A configuration dict, as returned by ``get_config``/``get_default_config``,
            optionally with overrides applied.
        results_file: If given, also save the waveform results to this path (``.pybert_data``).
    """
    pybert = PyBERT(run_simulation=False, gui=False)
    _apply_config(pybert, config)
    # update_plots populates pybert.plotdata, which save_results() needs; skip it otherwise.
    pybert.simulate(initial_run=True, update_plots=bool(results_file))

    if results_file:
        pybert.save_results(Path(results_file))

    return {name: _jsonify(getattr(pybert, name)) for name in RESULT_TRAIT_NAMES}


def inspect_results_file(results_file: str) -> dict:
    """Report the waveform arrays stored in a saved ``.pybert_data`` file.

    Note: these files only ever hold waveform arrays (impulse/step/frequency responses, etc.),
    never scalar performance metrics (BER, jitter, eye height/width) — those are only available
    live, right after a simulation runs, via ``run_simulation``'s return value.
    """
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    summary = {}
    for name, arr in results.the_data.arrays.items():
        info: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.dtype != object and arr.size:
            info.update(min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()))
        summary[name] = info
    return summary


def list_ibis_models(ibis_file: str) -> dict:
    """List the components and models defined in an IBIS file."""
    model = IBISModel(ibis_file, debug=False, gui=False)
    return {
        "components": list(model.model_dict["components"].keys()),
        "models": list(model.model_dict["models"].keys()),
        "parsing_errors": model.ibis_parsing_errors,
    }


def _safe_attr(obj: Any, name: str) -> Any:
    """`getattr`, tolerating attributes that only exist for some model types (e.g. `zout` is
    driver-only, `zin` is receiver-only, on `pyibisami`'s `Model` class)."""
    try:
        return getattr(obj, name)
    except AttributeError:
        return None


def inspect_ibis_model(ibis_file: str, model_name: str) -> dict:
    """Report the parameters of a single model from an IBIS file."""
    ibis_model = IBISModel(ibis_file, debug=False, gui=False)
    model = ibis_model.model_dict["models"][model_name]
    return {
        "type": model.mtype,
        "is_ami": model.is_ami,
        "zout": _jsonify(_safe_attr(model, "zout")),
        "zin": _jsonify(_safe_attr(model, "zin")),
        "ccomp": _jsonify(_safe_attr(model, "ccomp")),
        "slew": _jsonify(_safe_attr(model, "slew")),
        "ami_files": model.ami_files,
        "test_configs": list(model.test_configs.keys()) if model.is_ami else [],
        "summary": str(model),
    }


def build_server():
    """Construct the MCP server and register all PyBERT tools on it."""
    from mcp.server import MCPServer

    server = MCPServer("pybert", version=VERSION)
    for tool in (
        get_default_config,
        get_config,
        set_config,
        run_simulation,
        inspect_results_file,
        list_ibis_models,
        inspect_ibis_model,
    ):
        server.add_tool(tool)
    return server
