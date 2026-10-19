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
from pybert.models.bert import MIN_BATHTUB_VAL
from pybert.pybert import PyBERT
from pybert.threads.optimization import coopt
from pybert.utility.math import make_bathtub, make_vert_bathtubs
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
    "pjDD_dfe",
    "rjDD_dfe",
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


def _describe_ami_tuners(tuners: list) -> list[dict]:
    return [
        {
            "name": t.name,
            "min_val": t.min_val,
            "max_val": t.max_val,
            "step": t.step,
            "is_int": t.is_int,
            "enabled": t.enabled,
        }
        for t in tuners
    ]


def list_ami_params(config: dict) -> dict:
    """List the Tx/Rx IBIS-AMI ``Model_Specific`` parameters available for EQ optimization
    -- i.e. the names ``optimize_eq``'s ``tx_ami_params``/``rx_ami_params`` accept -- given a
    configuration's Tx/Rx IBIS/AMI file selections. Cheap: no simulation or optimization runs.

    Args:
        config: A configuration dict, as returned by ``get_config``/``get_default_config``.

    Returns:
        ``{"tx": [...], "rx": [...]}``, each a list of per-parameter dicts (``name``,
        ``min_val``, ``max_val``, ``step``, ``is_int``, ``enabled``). ``enabled`` reflects
        each side's IBIS-AMI file default (normally all ``False``: opt-in only), before any
        ``optimize_eq`` call. Empty on a side that isn't IBIS-AMI-active (``tx_use_ami``/
        ``rx_use_ami`` false) or has no ``Model_Specific`` "Range"-format parameters.
    """
    pybert = PyBERT(run_simulation=False, gui=False)
    _apply_config(pybert, config)
    return {
        "tx": _describe_ami_tuners(pybert.tx_ami_tap_tuners),
        "rx": _describe_ami_tuners(pybert.rx_ami_tap_tuners),
    }


def _select_ami_params(tuners: list, names: list[str], side: str) -> None:
    "Enable exactly the named tuners (by `.name`), disabling the rest; reject unknown names."
    tuner_names = {t.name for t in tuners}
    unknown = set(names) - tuner_names
    if unknown:
        raise ValueError(
            f"Unknown {side} AMI parameter name(s): {sorted(unknown)}. "
            f"Available: {sorted(tuner_names)}"
        )
    wanted = set(names)
    for tuner in tuners:
        tuner.enabled = tuner.name in wanted


def optimize_eq(
    config: dict,
    max_trials: int | None = None,
    tx_ami_params: list[str] | None = None,
    rx_ami_params: list[str] | None = None,
) -> dict:
    """Co-optimize Tx/Rx linear equalization for a given architecture, and return the result.

    Runs the same search `coopt()` drives for the GUI's "Optimize EQ" button (a TPE/Optuna
    search when Tx and/or Rx is IBIS-AMI, since each trial is a real ``AMI_Init()`` call;
    exact closed-form MMSE or a small exhaustive grid otherwise) -- scored by a cheap
    pulse-response figure of merit, not a full bit-level simulation. Call ``run_simulation``
    afterward, on this call's returned ``config``, to get the real BER/jitter metrics for
    the winning EQ. Architecture (tap counts, CTLE enable, etc.) should already be set in
    ``config``; this only searches EQ *values*.

    Note: native Rx-FFE is a *static* filter (`rx_taps[k].value`, applied as-is -- it does
    not adapt inside `run_simulation()`), so this function's returned `config` does carry
    its winning weights forward, just like Tx-FFE weights, CTLE peaking, and AMI parameter
    values. DFE, by contrast, *is* trained adaptively (LMS) inside every `run_simulation()`
    call, starting fresh from zero each time regardless of any prior value -- so its tap
    weights aren't part of the persisted configuration schema (only their enabled/min/max
    bounds are), and this function's committed `dfe_tap_tuners[k].value` has no effect on
    a later `run_simulation()` call. Both are still reported below, for visibility into
    what this search found.

    Args:
        config: A configuration dict, as returned by ``get_config``/``get_default_config``,
            optionally with overrides applied.
        max_trials: Override the number of Optuna trials run when Tx and/or Rx is IBIS-AMI
            (each trial invokes a real ``AMI_Init()`` call, so runtime scales with this);
            PyBERT's own ``ami_opt_trials`` default (normally 100) otherwise. Not part of
            the persisted config schema, so it isn't carried in ``config`` -- pass it fresh
            on every call. Ignored when neither side is AMI-active (native EQ is an exact
            closed-form/grid solve, not a trial search).
        tx_ami_params: Names (from ``list_ami_params``) of Tx IBIS-AMI parameters to include
            in the search; every other Tx AMI parameter is disabled. ``None`` (default)
            leaves each parameter's enabled state as the IBIS-AMI file set it (normally all
            disabled -- opt-in only), which usually means nothing is actually searched.
            Ignored when Tx isn't IBIS-AMI-active.
        rx_ami_params: Same as ``tx_ami_params``, for the Rx side.

    Returns:
        A dict with the winning Tx/Rx tap weights, CTLE peaking, AMI parameter values, the
        figure of merit, and ``config``: the input config updated with the winning native
        Tx-FFE/CTLE/AMI values, ready to pass to ``run_simulation``.

    Raises:
        ValueError: If `max_trials` isn't a positive integer, or `tx_ami_params`/
            `rx_ami_params` names a parameter that side's IBIS-AMI file doesn't declare.
        RuntimeError: If the optimization fails to converge.
    """
    if max_trials is not None and max_trials < 1:
        raise ValueError(f"max_trials must be a positive integer, not {max_trials!r}")

    pybert = PyBERT(run_simulation=False, gui=False)
    _apply_config(pybert, config)
    if max_trials is not None:
        pybert.ami_opt_trials = max_trials
    if tx_ami_params is not None:
        _select_ami_params(pybert.tx_ami_tap_tuners, tx_ami_params, "Tx")
    if rx_ami_params is not None:
        _select_ami_params(pybert.rx_ami_tap_tuners, rx_ami_params, "Rx")

    tx_weights, rx_peaking, rx_weights, fom, valid, tx_ami_vals, rx_ami_vals = coopt(pybert)
    if not valid:
        raise RuntimeError("EQ optimization failed to converge.")

    # coopt() already commits DFE tap weights and, when AMI-active, the winning AMI
    # parameter values onto the live traits `run_simulation` reads. Native Tx/Rx linear
    # EQ needs the same "Use EQ" commit the GUI's Ctrl+U action performs (see
    # `MyHandler.do_use_eq` in src/pybert/gui/handler.py): each tap takes its tuner's weight
    # *and* enabled state, since the simulation ignores a disabled tap's weight -- coopt() searches
    # the tuners' enabled taps, which needn't match the taps' own enabled flags.
    if not pybert.tx_use_ami:
        for i, weight in enumerate(tx_weights):
            pybert.tx_taps[i].value = weight
            pybert.tx_taps[i].enabled = pybert.tx_tap_tuners[i].enabled
    if not pybert.rx_use_ami:
        pybert.peak_mag = rx_peaking
        pybert.peak_freq = pybert.peak_freq_tune
        pybert.rx_bw = pybert.rx_bw_tune
        pybert.ctle_enable = pybert.ctle_enable_tune
        for i, weight in enumerate(rx_weights):
            pybert.rx_taps[i].value = weight
            pybert.rx_taps[i].enabled = pybert.ffe_tap_tuners[i].enabled

    return {
        "fom": fom,
        "tx_weights": _jsonify(tx_weights),
        "rx_peaking": rx_peaking,
        "rx_weights": _jsonify(rx_weights),
        "tx_ami_vals": _jsonify(tx_ami_vals),
        "rx_ami_vals": _jsonify(rx_ami_vals),
        "config": _config_dict(pybert),
    }


def _jitter_bathtub_width_ui(pybert: PyBERT, target_ber: float) -> float | None:
    """Post-DFE *crossing-jitter* bathtub width at `target_ber` (fraction of a UI), via PyBERT's
    own bathtub curve (`make_bathtub()`) over its own measured jitter histogram, its tails
    extrapolated with the separated Rj from the bounded jitter's extent -- the same computation
    and data (`jitter_bins`, `jitter_dfe`, `rj_dfe`, `dj_pos_dfe`, `dj_neg_dfe`, `ui`) the GUI's
    own bathtub plot uses. (The dual-Dirac fit's values are for reference only.) Requires the
    preceding `pybert.simulate()` call to have used `update_plots=True` (that's what populates
    the jitter-histogram traits this reads).

    This is a timing-only measure: the spread of 0-V crossing times (for PAM-4, of the
    transitions symmetric about 0 V only). It can't see voltage ISI or noise at the sampling
    instant, symbols that never cross 0 V, or PAM-4's upper/lower eyes, so it can report an
    open eye where the real eye is closed; `_eye_width_ui()` is the eye opening.

    Returns None if the curve doesn't reach `target_ber` anywhere.
    """
    bathtub = make_bathtub(
        pybert.jitter_bins, pybert.jitter_dfe, min_val=0.1 * MIN_BATHTUB_VAL,
        rj=pybert.rj_dfe, mu_r=pybert.dj_pos_dfe, mu_l=pybert.dj_neg_dfe, extrap=True,
    )
    bins = np.asarray(pybert.jitter_bins)
    bt = np.asarray(bathtub)
    center = len(bt) // 2
    if bt[center] > target_ber:
        return None

    # bt is monotonically decreasing left-of-center, increasing right-of-center (a real
    # bathtub shape); np.interp needs an ascending x, hence the reversal on the left half.
    left_cross = np.interp(target_ber, bt[:center + 1][::-1], bins[:center + 1][::-1])
    right_cross = np.interp(target_ber, bt[center:], bins[center:])
    return float((right_cross - left_cross) / pybert.ui)


def _eye_heights_at(samps: np.ndarray, decisions: np.ndarray, target_ber: float) -> list[float | None]:
    """Vertical opening (V) of each eye at `target_ber`, lowest eye first, via PyBERT's own vertical
    bathtub curves (`make_vert_bathtubs()`) over the given slicer-input samples and the symbol
    decisions labelling them. An eye that doesn't reach `target_ber` anywhere reports None.
    """
    heights: list[float | None] = []
    for v, bt in make_vert_bathtubs(samps, decisions, min_val=0.1 * MIN_BATHTUB_VAL):
        best = int(np.argmin(bt))
        if bt[best] > target_ber:
            heights.append(None)
            continue
        # bt falls from the lower level's mean to its minimum, then rises to the upper level's.
        lo_cross = np.interp(target_ber, bt[:best + 1][::-1], v[:best + 1][::-1])
        hi_cross = np.interp(target_ber, bt[best:], v[best:])
        heights.append(float(hi_cross - lo_cross))
    return heights


def _vert_eye_heights_v(pybert: PyBERT, target_ber: float) -> list[float | None]:
    """Real post-DFE vertical eye openings at `target_ber` (V), one per eye, lowest eye
    first, via PyBERT's own vertical bathtub curves (`make_vert_bathtubs()`) over the DFE
    slicer input samples PyBERT itself took at its recovered clock instants.

    An eye that doesn't reach `target_ber` anywhere reports None (as `eye_margin_ui` does).
    """
    return _eye_heights_at(pybert.dfe_slicer_samps, pybert.dfe_slicer_decisions, target_ber)


def _eye_width_ui(pybert: PyBERT, target_ber: float) -> float | None:
    """Real post-DFE eye width at `target_ber` (fraction of a UI): the span of sampling offsets
    around PyBERT's recovered clock over which *every* eye's vertical opening at `target_ber`
    stays positive -- the 2-D eye's width at that BER.

    At each offset, the DFE summing-node output is sampled at the slicer's own clock instants
    shifted by that many time steps, those samples are labelled by the decisions made at the
    clock, and their openings come from the same `make_vert_bathtubs()` computation
    `eye_heights_v` uses (which is this at offset zero). Offsets stay within the slicer's own
    UI (the DFE's feedback switches at its edges), so the resolution is ``1 / nspui`` UI and the
    largest reportable width ``(nspui - 1) / nspui``.

    Returns None if the eye is closed at the clock itself (the receiver samples a closed eye).
    """
    y = np.asarray(pybert.dfe_out)
    decisions = np.asarray(pybert.dfe_slicer_decisions)
    n_samps = len(decisions)
    clock_ixs = np.where(np.asarray(pybert.clocks))[0][-n_samps:]
    if len(clock_ixs) != n_samps or not np.allclose(y[clock_ixs], pybert.dfe_slicer_samps):
        raise RuntimeError("DFE output at the recovered clock indices doesn't match the slicer's samples!")
    nspui = pybert.nspui
    half = nspui // 2

    def is_open(ofst: int) -> bool:
        ixs = clock_ixs + ofst
        keep = (ixs >= 0) & (ixs < len(y))
        return all(h is not None and h > 0 for h in _eye_heights_at(y[ixs[keep]], decisions[keep], target_ber))

    if not is_open(0):
        return None
    lo = 0
    while lo - 1 > -half and is_open(lo - 1):
        lo -= 1
    hi = 0
    while hi + 1 < half and is_open(hi + 1):
        hi += 1
    return (hi - lo + 1) / nspui


def run_simulation(config: dict, results_file: str | None = None,
                    target_ber: float | None = None) -> dict:
    """Run a headless PyBERT simulation and return its summary performance metrics.

    Args:
        config: A configuration dict, as returned by ``get_config``/``get_default_config``,
            optionally with overrides applied.
        results_file: If given, also save the waveform results to this path (``.pybert_data``).
        target_ber: If given, also return, at this BER, the real post-DFE eye width
            (UI; see `_eye_width_ui`) as ``eye_margin_ui`` -- ``None`` if the eye is closed
            at the recovered clock -- the vertical opening of each eye (V, lowest eye first;
            see `_vert_eye_heights_v`) as ``eye_heights_v``, and the crossing-jitter bathtub
            width (UI; timing only, see `_jitter_bathtub_width_ui`) as ``jitter_bathtub_width_ui``.
    """
    pybert = PyBERT(run_simulation=False, gui=False)
    _apply_config(pybert, config)
    # update_plots populates pybert.plotdata (needed by save_results()) and the jitter
    # histogram traits (needed for target_ber's jitter bathtub); skip it otherwise.
    update_plots = bool(results_file) or target_ber is not None
    pybert.simulate(initial_run=True, update_plots=update_plots)

    if results_file:
        pybert.save_results(Path(results_file))

    result = {name: _jsonify(getattr(pybert, name)) for name in RESULT_TRAIT_NAMES}
    if target_ber is not None:
        result["eye_margin_ui"] = _eye_width_ui(pybert, target_ber)
        result["eye_heights_v"] = _vert_eye_heights_v(pybert, target_ber)
        result["jitter_bathtub_width_ui"] = _jitter_bathtub_width_ui(pybert, target_ber)
    return result


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
    model = IBISModel(ibis_file)
    return {
        "components": list(model.model_dict["components"].keys()),
        "models": list(model.model_dict["models"].keys()),
        "parsing_errors": model.ibis_parsing_errors,
    }


def _platform_ami_files(ibis_model: IBISModel, model: Any) -> tuple[Any, Any]:
    """The model's executable and AMI files for this platform, resolved against the IBIS file's directory
    (`None` if none). `I/O` models name them per direction, so they're given as `{"tx": ..., "rx": ...}`."""

    def resolved(is_tx: bool | None) -> tuple[str | None, str | None]:
        return tuple(  # type: ignore[return-value]
            str(path) if (path := ibis_model.resolve_file(name)) else None for name in model.get_ami_files(is_tx)
        )

    if model.is_io:
        (tx_dll, tx_ami), (rx_dll, rx_ami) = resolved(True), resolved(False)
        return {"tx": tx_dll, "rx": rx_dll}, {"tx": tx_ami, "rx": rx_ami}
    return resolved(None)


def inspect_ibis_model(ibis_file: str, model_name: str) -> dict:
    """Report the parameters of a single model from an IBIS file."""
    ibis_model = IBISModel(ibis_file)
    model = ibis_model.model_dict["models"][model_name]
    dll_file, ami_file = _platform_ami_files(ibis_model, model)
    return {
        "type": model.mtype,
        "is_ami": model.is_ami,
        "zout": _jsonify(model.zout),  # `None` when not applicable (e.g. - `zout` of an `Input` model).
        "zin": _jsonify(model.zin),
        "ccomp": _jsonify(model.ccomp),
        "slew": _jsonify(model.slew),
        "ami_files": model.ami_files,
        "dll_file": dll_file,
        "ami_file": ami_file,
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
        list_ami_params,
        optimize_eq,
        run_simulation,
        inspect_results_file,
        list_ibis_models,
        inspect_ibis_model,
    ):
        server.add_tool(tool)
    return server
