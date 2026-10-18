# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Trust and reuse PyBERT's own calculations

When a task needs a derived quantity (margin, BER, jitter decomposition, eye opening,
channel delay, ...), check whether PyBERT already computes it before writing a new formula
or approximation for it. PyBERT's own signal-processing code (`src/pybert/utility/math.py`,
`src/pybert/models/bert.py`) reflects real domain expertise and is validated against this
project's own GUI and test suite; a hand-rolled approximation written from outside that code
is a second, unverified implementation that can silently diverge from the real answer by a
large margin — not a shortcut. Concrete case from this project's own history: an
`eye_margin_ui()` helper (in the local architecture-search driver) approximated eye margin as
a simple additive dual-Dirac jitter budget, computed from PyBERT's raw jitter scalars.
It reported a badly closed eye (-1.155 UI) for a configuration that PyBERT's own GUI
confirmed produces zero real bit errors and, via its actual bathtub-curve extrapolation, a
genuinely open +0.15 UI eye at the same target BER — a wrong-sign, not just imprecise,
result. Replacing it with a function that calls PyBERT's own `make_bathtub()` over PyBERT's
own measured jitter histogram (`src/pybert/mcp/server.py`'s `run_simulation(...,
target_ber=...)`) matched the verified GUI result almost exactly. (That crossing-jitter bathtub
later proved to be timing-only -- blind to voltage ISI at the sampler, so it can call a closed
PAM-4 eye open -- and `eye_margin_ui` is now the 2-D eye's width from PyBERT's own
`make_vert_bathtubs()`; see the MCP section below.)

Apply this at all costs when working in this repo:

- Before computing something PyBERT already exposes as a trait (`pybert.ui`, `pybert.chnl_dly`,
  `pybert.n_errs_dfe`, `pybert.isi_dfe`/`dcd_dfe`/`pj_dfe`/`rj_dfe`/`pjDD_dfe`/`rjDD_dfe`, ...),
  read the trait — don't re-derive it independently.
- Before writing a formula that approximates something PyBERT already computes exactly
  (jitter budgets, bathtub curves, BER, channel responses), call PyBERT's own function on
  PyBERT's own data instead.
- If a needed value isn't currently exposed via the MCP server or elsewhere, the right fix is
  usually to expose it (add the trait to `RESULT_TRAIT_NAMES`, or add a small new MCP tool
  that wraps PyBERT's own computation) — not to write an external approximation of it.
- Don't re-run a simulation to obtain a value derivable from a simulation already run; fold
  additional requested outputs (e.g. a margin at some `target_ber`) into the *same*
  `run_simulation()` call rather than issuing a second one.

## Commands

All non-`make` commands must be prefixed with `uv run`.
The `Makefile` handles multi-version testing via `uv`.

```bash
# Run the full test suite (current Python only, fastest)
uv run pytest tests/

# Run a single test
uv run pytest tests/test_basic.py::TestBasic::test_status

# Run tests across all supported Python versions (3.10, 3.11, 3.12)
make test

# Lint (ruff + flake8, line-length 119)
make lint

# Type check (mypy)
make type-check

# Build package
make build

# Start the MCP server (requires: uv sync --extra mcp)
uv run pybert mcp
```

## Architecture

PyBERT is a serial-link BERT simulator built on the Enthought Traits/TraitsUI stack. It is a single `HasTraits` class (`PyBERT` in `src/pybert/pybert.py`) that acts simultaneously as the data model, the simulation controller, and the GUI model.

### Simulation data flow

1. **Entry**: `PyBERT.simulate()` → `my_run_simulation()` in `src/pybert/models/bert.py`. In GUI mode this runs in a background `RunSimThread` (see `src/pybert/threads/sim.py`).
2. **Channel construction** (`bert.py` ~line 1300): Either uses Howard Johnson's UTP model (default) or imports a channel from file(s) via `import_channel()` in `src/pybert/utility/sparam.py`. When multiple files are listed in `self.ch_files`, they are cascaded using `skrf`'s `**` (network cascade) operator. `import_channel()`/`import_freq()` accept 1/2/4/8/12-port Touchstone files natively; a VITA 68.x reference-model segment (Module or Backplane) instead bundles both link directions per lane into an 8-port block (e.g. 32 ports for 4 lanes) — pass `vita_end="tx"` or `"rx"` to select the right 4-port sub-network per lane (every segment in a cascade uses `"tx"` except the terminal Rx-side Module, which uses `"rx"` — see `import_freq()`'s docstring for the full port-offset derivation, verified against real VITA68.3 reference files). With `renumber=True`, 4-port files' through-path convention ("1=>2/3=>4" vs. "1=>3/2=>4") is detected by `thru_is_1to3()`, which compares the two candidate through pairs averaged over 10 MHz-1 GHz (where a through path is near 1 and intra-pair coupling near 0); the previous single mid-band comparison (5% of f_max) mis-detected long, lossy channels, whose through path can fall below that coupling there -- 7 of the 533 P802.3dj corpus victim files.
   **P802.3dj device models (`pkg_enable`)**: with `pkg_enable` true, `calc_chnl_h()` replaces the lumped
   `rs`/`cout` and `rin`/`cin` (applied by `utility/channel.py::calc_G()`, which ignores the channel's S11/S22) with
   the P802.3dj/D3.3 Annex 178A transmitter/receiver models -- 3-stage C_d/L_s die ladder + C_b, package
   transmission-line stages + C_p, optional partial host channel -- from PyChOpMarg (`pychopmarg.devmodel`,
   per-group presets in `pychopmarg.config.ieee_8023dj_d3p3`): `calc_pkg_chnl_H()` renormalizes the channel to
   2 x R0 (92.5 ohm), cascades Tx model ** channel ** Rx model, and terminates it in each device's R_d per (178A-3),
   halved to keep `chnl_H`'s driver-open-circuit-voltage reference. `pkg_group` ("KR"/"C2C"/"CR"/"C2M") selects
   the draft's table; `tx_pkg_case`/`rx_pkg_case` the package class + Test ("A1"/"A2"/"B1"/"B2"), CR host class
   ("HL"/"HN"/"HH"), or C2M "host"/"module"; `pkg_include_host` adds the partial host PCB (leave off for channel
   data that already includes it). Native Tx/Rx only (IBIS-AMI on-die additions are ignored on this path). A bare
   matched termination reproduces `calc_G()` exactly (`tests/test_pkg_models.py`).
3. **IBIS-AMI integration**: `src/pybert/utility/ibisami.py::run_ami_model()` wraps `pyibisami.ami.model.AMIModel`. The response dict returned by `AMIModel.get_responses()` is keyed with `AmiModelResponseKey` objects (frozen dataclass) — always use the exported constants `IMP_RESP_INIT`, `IMP_RESP_GETW`, `OUT_RESP_INIT`, `OUT_RESP_GETW` when accessing it, not plain strings.
4. **Output**: Results are stored as Traits attributes on the `PyBERT` instance; Chaco plots are updated via `plotdata` (an `ArrayPlotData`).

### Key files

| File | Role |
|------|------|
| `src/pybert/pybert.py` | `PyBERT` class: all traits, button handlers, `simulate()`, IBIS file-change observers |
| `src/pybert/models/bert.py` | `my_run_simulation()` — the full simulation pipeline |
| `src/pybert/gui/view.py` | TraitsUI view definition (`traits_view`) |
| `src/pybert/gui/handler.py` | `MyHandler` — toolbar/menu actions (save/load config, run/stop sim) |
| `src/pybert/configuration.py` | `PyBertCfg` — YAML/pickle config serialisation |
| `src/pybert/utility/sparam.py` | S-parameter helpers: `import_channel`, `import_freq` (incl. VITA 68.x `vita_end` lane extraction), `sdd_21`, `interp_s2p` |
| `src/pybert/utility/ibisami.py` | `run_ami_model()` — drives pyibisami DLL/SO |
| `src/pybert/utility/sigproc.py` | Signal processing: `import_time`, `trim_impulse`, `raised_cosine` |
| `src/pybert/mcp/server.py` | MCP server (`pybert mcp`, needs the `mcp` extra): tools for running simulations, co-optimizing Tx/Rx EQ, listing IBIS-AMI optimizer parameters, reading/writing config, browsing saved results, and inspecting IBIS-AMI models |

### MCP server (`src/pybert/mcp/`)

`pybert mcp` starts a stdio MCP server (built on the `mcp` package, an optional extra —
`pip install "pipbert[mcp]"` / `uv sync --extra mcp`). `src/pybert/mcp/server.py` exposes plain,
directly-testable functions (`run_simulation`, `optimize_eq`, `list_ami_params`,
`get_config`/`set_config`, `inspect_results_file`, `list_ibis_models`/`inspect_ibis_model`) and
wraps them as MCP tools in `build_server()`. `optimize_eq()` drives
`pybert.threads.optimization.coopt()` — the same TPE(Optuna)/closed-form-MMSE search behind the
GUI's "Optimize EQ" button — against a headless `PyBERT` instance, scored by a cheap
pulse-response figure of merit (not a full bit-level sim); its returned `config` carries the
winning Tx-FFE/CTLE/AMI values forward (replicating the GUI's "Use EQ" commit step,
`MyHandler.do_use_eq`), ready to pass straight to `run_simulation`. An optional `max_trials` arg
caps the Optuna trial budget on the AMI path (each trial is a real `AMI_Init()` call, so runtime
scales with it) by overriding `ami_opt_trials` directly on the headless instance —
`ami_opt_trials` isn't part of the persisted config schema, so this is a call-level override, not
something carried in `config`. `coopt()`'s native Tx search skips tap combinations whose cursor
(1 - sum of the other taps' magnitudes) is <= 0 (an inverted pulse) or below the `tx_c0_min` trait (default 0;
P802.3dj's reference transmitter uses 0.54). `tx_ami_params`/`rx_ami_params` (lists of parameter names) pick
which IBIS-AMI `Model_Specific` parameters actually get searched — enabled-state for
`tx_ami_tap_tuners`/`rx_ami_tap_tuners` isn't in the persisted config schema either and defaults
to all-disabled (opt-in only), so without these an AMI-active `optimize_eq()` call degenerates to
a single fixed evaluation; `list_ami_params(config)` (no simulation/optimization run) reports each
side's available parameter names/ranges so callers know what to pass. Note also that a config's
`rx_use_ami`/`tx_use_ami` fields are read-only side effects, not real switches — setting
`rx_ibis_file` resets `rx_use_ami` to `False` as part of (re-)parsing it; the actual switch is
`rx_eq_sel = "IBIS-AMI"` (mirror on Tx), applied once the file's parsed valid (see
`_rx_ibis_file_changed`/`_rx_eq_sel_changed` in `pybert.py`) — the same trap a caller hits trying
to enable AMI in `config` for `run_simulation`, not just for `optimize_eq()`. Similarly,
`ch_files` alone does nothing — it's read only when `inter_sel` is also set to `"multiple"`
(or `ch_file`/`"single"` for one file); `inter_sel` defaults to `"native"` (the analytic Howard
Johnson model), which silently ignores `ch_files`/`ch_file` if left unset. Native Rx-FFE
(`rx_taps`) is a *static* filter — it never adapts inside `run_simulation()`, unlike DFE
(LMS-adaptive, starting fresh from zero regardless of any prior value, except tap 1 under the Mueller-Muller CDR) — so `rx_taps`
*is* part of the persisted config schema (`(enabled, value, min_val, max_val)` per tap,
mirroring `tx_taps`; `ffe_tap_tuners`, its EQ-tune-tab counterpart, round-trips too), while
DFE tap *weights* deliberately aren't (only their enabled/min/max bounds are — a `value`
would be silently ignored anyway). Getting native Rx-FFE weights into a config — via
`optimize_eq()`, `coopt()` + `save_configuration()`, or hand-editing — is required for a
realistic `run_simulation()`/GUI result: with no such weights set, `rx_taps` defaults to a
trivial identity pass-through (one unity-gain tap, everything else zero), which can look
like a channel/architecture failure that's actually just an unequalized Rx path. Each call
constructs its own fresh `PyBERT(gui=False)` instance; nothing is cached across calls.
`_apply_config()` round-trips config dicts through a temp YAML file and `PyBERT.load_configuration()`
so `PyBertCfg`'s existing tap-tuple/legacy-rename handling is reused rather than duplicated. Note:
`.pybert_data` files only ever hold waveform arrays, never scalar performance metrics — those are
only available live, right after `simulate()`, which is why `run_simulation` returns them directly.
`run_simulation`'s optional `target_ber` returns the *real* post-DFE eye width at that BER as
`eye_margin_ui` (UI; `None` if the eye is closed at the recovered clock): the span of sampling
offsets around the clock over which *every* eye's vertical opening at `target_ber` stays positive,
each offset scored by the same `make_vert_bathtubs()` computation as `eye_heights_v` (below), over
the DFE summing-node output (`dfe_out`) at the slicer's own clock indices (`clocks`) shifted by
that many time steps, labelled by the decisions made at the clock (resolution `1/nspui` UI, at most
`(nspui - 1)/nspui`). It also returns `jitter_bathtub_width_ui`: the crossing-jitter bathtub's width,
from PyBERT's own `make_bathtub()` (`src/pybert/utility/math.py`) over its measured jitter histogram
(`jitter_bins`/`jitter_dfe`), tails extrapolated with the separated Rj (`rj_dfe`) from the bounded
jitter's extent (`dj_pos_dfe`/`dj_neg_dfe`) -- what the GUI's bathtub plot shows. That one is
timing-only (0-V crossing times; for PAM-4, symmetric transitions only), blind to voltage ISI/noise
at the sampler and to PAM-4's upper/lower eyes: on li_dj with a bang-bang CDR it reported 0.36 UI
open while the real eye was closed at every offset (96 errors), and it overstates even open eyes
(0.28 UI vs. a measured 0.20 UI worst-case width). Use `eye_margin_ui`/`eye_heights_v` to judge an eye.
The dual-Dirac values (`pjDD_*`, `rjDD_*`,
`mu_pos_*`/`mu_neg_*`) are for reference only: don't build further calculations on them. The
separated Rj needs at least two pattern repetitions in the `eye_bits` window (e.g. PAM-4 PRBS-15 is
65,534 UI long; PRBS-13 is 16,382 UI); with fewer, it reads 0 and bathtubs aren't extrapolated.
For PAM-4, jitter statistics use only transitions symmetric about 0 V (see `calc_jitter()`'s
`xing_mask`). Passing `target_ber` makes this call use
`update_plots=True` internally (needed to populate the jitter-histogram traits for
`jitter_bathtub_width_ui`), same as passing
`results_file`. The same call also returns `eye_heights_v`: each eye's vertical opening (V, lowest
eye first; `None` per closed eye) at `target_ber`, from `make_vert_bathtubs()` over the DFE slicer
input samples PyBERT took at its own recovered clock instants (`dfe_slicer_samps`/
`dfe_slicer_decisions`, stored by `my_run_simulation()` over the same `eye_uis` window
`calc_ber()` checks). Each level's tail is measured where >= 10 samples lie beyond, and
extrapolated past that with a Gaussian fit to its 100 most extreme samples on the
normal-probability (Q) scale -- *not* by a Gaussian fitted to each level's histogram
(`curve_fit(gaus_pdf, ...)` over its below-half-maximum region, in volts), nor by a fit over
the whole tail: residual ISI makes level distributions sub-Gaussian (bounded) in their tails,
and both of those overstated them badly, falsely closing eyes that a 600k-symbol run
measured open. With short runs (~1k samples per level) the extrapolation is
still conservative, since the tail's steepening toward the ISI bound isn't yet visible;
it converges on the measured openings as `nbits`/`eye_bits` grow.

### Submodule: PyAMI (`PyAMI/`)

A git submodule (also a `uv` workspace member). Contains the `pyibisami` package. Its own `CLAUDE.md` at `PyAMI/CLAUDE.md` documents it in detail. Key invariants for PyBERT callers:

- `IBISModel(ibis_file_name)` — **one-argument form only** (pyibisami >= 10.0 removed `debug`/`gui`). It never opens dialogs itself; user-facing notices fire via its `alert` event (`Event(Str)`) and go to the `pyibisami` logger. `PyBERT` observes `alert` (`_log_ibis_alert`) and emits its own load notice (`_ibis_load_notice`), since notices fired during construction can't be observed.
- PyBERT depends on `pyibis-ami[gui]`: `AMIParamConfigurator.open_gui()`, calling `IBISModel`/`Model` instances, and `Model.plot_iv` need the `gui` extra.
- `AMIModel.get_responses()` returns `dict[AmiModelResponseKey, ...]` — access with the module-level constants, not strings.

### Traits/TraitsUI patterns

- All independent variables are class-level `Trait` declarations on `PyBERT`.
- Derived/computed values use `Property(..., depends_on=[...])` + `@cached_property` getters named `_get_<trait_name>`.
- Button traits (`Button(label="...")`) get `_<trait_name>_fired()` handlers on the class.
- GUI is constructed in `view.py`; test code always uses `PyBERT(gui=False)`.
- `enabled_when=` expressions in the view are Python expressions evaluated against the model's trait values.

### Configuration persistence

`PyBertCfg` (in `configuration.py`) is pickled/YAML-dumped. `load_from_file` iterates `vars(user_config)` and calls `setattr(pybert, prop, value)` for most props. Special-cased props (tap lists) are handled explicitly. Old configs without `ch_files` fall back gracefully: the simulation reads `self.ch_file` (single path) when `self.ch_files` is empty. The CDR's proportional step is `delta_t_ui` (in UI, so loop dynamics don't change with data rate); old configs' absolute `delta_t` (ps) is converted on load using that config's own UI. `cdr_type` selects the CDR:
`"Bang-Bang"` (default; the adaptive loop in `models/cdr.py`) or `"Mueller-Muller"`, which clocks the DFE at a fixed
phase (nominal UI, locked from the start) found by PyChOpMarg's `loc_curs()` on the slicer-input pulse response
(`ffe_out_p`), anticipating DFE tap 1 within its limits -- the same criterion COM uses to locate its cursor. It models
the MM loop's lock point, not its dynamics; since that point presumes DFE tap 1 has converged to the anticipated value
(93A-26), the MM case starts tap 1 there (the one exception to the DFE starting from zero) -- from zero, the
decision-directed LMS can't bootstrap when the pre-DFE eye at that phase is closed (e.g. a 0.4 post-cursor on PAM-4).

### CLI entry point

`pybert` CLI is defined in `src/pybert/cli.py`. `pybert` (no subcommand) opens the GUI; `pybert sim <config.yaml>` runs headless and writes `.pybert_data`.
