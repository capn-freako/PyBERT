# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
```

## Architecture

PyBERT is a serial-link BERT simulator built on the Enthought Traits/TraitsUI stack. It is a single `HasTraits` class (`PyBERT` in `src/pybert/pybert.py`) that acts simultaneously as the data model, the simulation controller, and the GUI model.

### Simulation data flow

1. **Entry**: `PyBERT.simulate()` → `my_run_simulation()` in `src/pybert/models/bert.py`. In GUI mode this runs in a background `RunSimThread` (see `src/pybert/threads/sim.py`).
2. **Channel construction** (`bert.py` ~line 1300): Either uses Howard Johnson's UTP model (default) or imports a channel from file(s) via `import_channel()` in `src/pybert/utility/sparam.py`. When multiple files are listed in `self.ch_files`, they are cascaded using `skrf`'s `**` (network cascade) operator.
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
| `src/pybert/utility/sparam.py` | S-parameter helpers: `import_channel`, `import_freq`, `sdd_21`, `interp_s2p` |
| `src/pybert/utility/ibisami.py` | `run_ami_model()` — drives pyibisami DLL/SO |
| `src/pybert/utility/sigproc.py` | Signal processing: `import_time`, `trim_impulse`, `raised_cosine` |

### Submodule: PyAMI (`PyAMI/`)

A git submodule (also a `uv` workspace member). Contains the `pyibisami` package. Its own `CLAUDE.md` at `PyAMI/CLAUDE.md` documents it in detail. Key invariants for PyBERT callers:

- `IBISModel(ibis_file_name, debug=False, gui=True)` — **two-argument form only**; a removed third positional argument (`show_ui`) caused bugs; always pass `debug` and `gui` as keyword args.
- `AMIModel.get_responses()` returns `dict[AmiModelResponseKey, ...]` — access with the module-level constants, not strings.

### Traits/TraitsUI patterns

- All independent variables are class-level `Trait` declarations on `PyBERT`.
- Derived/computed values use `Property(..., depends_on=[...])` + `@cached_property` getters named `_get_<trait_name>`.
- Button traits (`Button(label="...")`) get `_<trait_name>_fired()` handlers on the class.
- GUI is constructed in `view.py`; test code always uses `PyBERT(gui=False)`.
- `enabled_when=` expressions in the view are Python expressions evaluated against the model's trait values.

### Configuration persistence

`PyBertCfg` (in `configuration.py`) is pickled/YAML-dumped. `load_from_file` iterates `vars(user_config)` and calls `setattr(pybert, prop, value)` for most props. Special-cased props (tap lists) are handled explicitly. Old configs without `ch_files` fall back gracefully: the simulation reads `self.ch_file` (single path) when `self.ch_files` is empty.

### CLI entry point

`pybert` CLI is defined in `src/pybert/cli.py`. `pybert` (no subcommand) opens the GUI; `pybert sim <config.yaml>` runs headless and writes `.pybert_data`.
