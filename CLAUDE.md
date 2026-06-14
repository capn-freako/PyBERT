# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What PyBERT Is

PyBERT is a serial communication link **bit error rate tester (BERT) simulator** with a GUI. It simulates a complete TX→Channel→RX signal chain including equalization (TX FFE, RX CTLE, DFE), clock/data recovery (CDR), jitter analysis, and optional IBIS-AMI model integration. The PyPI package name is `pipbert` (the name `pybert` was taken).

The repo contains three sub-projects:
- **`src/pybert/`** — the main PyBERT application (this repo's primary concern)
- **`PyAMI/`** — a git submodule (`pyibis-ami` package), used as a uv workspace member
- **`PyChOpMarg/`** — a git submodule (`pychopmarg` package), depended on for EQ optimization

## Commands

```bash
make test          # Run tests across Python 3.10, 3.11, 3.12
make lint          # ruff + flake8
make type-check    # mypy
make format        # isort + black (destructive — use caution)
make docs          # Sphinx → docs/build/index.html
make build         # Build source tarball and wheel
make check         # Validate pyproject.toml
make clean         # Remove build artifacts, .venv, caches
```

Run a single test file:
```bash
uv run pytest tests/test_basic.py -vv
```

Run with a specific Python version:
```bash
uv run --python 3.11 pytest tests/test_basic.py -vv
```

Run the GUI:
```bash
uv run pybert
```

Run headless simulation from config file:
```bash
uv run pybert sim path/to/config.yaml
```

## Architecture

### Core Class: `PyBERT` (`src/pybert/pybert.py`)

`PyBERT` is a `HasTraits` subclass — all simulation parameters are declared as Traits attributes (e.g., `bit_rate`, `nbits`, `nspui`). Traits drives reactive data binding between the model and the GUI. The GUI toolkit is PySide6 via the Enthought `traitsui`/`chaco`/`enable` stack.

Key `PyBERT` attributes:
- `bit_rate`, `nbits`, `nspui` — simulation control
- `ch_file`, `use_ch_file` — optional Touchstone S-parameter channel file
- `tx_tap_tuners` / `dfe_tap_tuners` — lists of `TxTapTuner` objects for EQ optimization
- `do_sweep` — when True, sweeps through multiple channel Touchstone models

### Simulation Pipeline (`src/pybert/models/bert.py`)

`my_run_simulation(pybert)` is the top-level simulation function. It runs in a dedicated thread (`RunSimThread`) to keep the GUI responsive. The pipeline:

1. Build or import channel impulse response (from parametric model or Touchstone file via `skrf`)
2. Apply TX FFE (pre/post cursor taps)
3. Convolve through channel
4. Apply RX CTLE (analog equalizer)
5. Run CDR (bang-bang, `src/pybert/models/cdr.py`)
6. Run DFE (decision feedback equalizer, `src/pybert/models/dfe.py`)
7. Optionally run Viterbi decoder (`src/pybert/models/viterbi.py`)
8. Compute jitter decomposition (ISI, DCD, PJ, RJ) and BER
9. Update Chaco plots

IBIS-AMI models can replace the built-in TX/RX EQ — see `src/pybert/utility/ibisami.py` and example models in `models/ibisami/`.

### EQ Optimization (`src/pybert/threads/optimization.py`)

`OptThread` runs co-optimization of TX FFE + RX CTLE/DFE using `pychopmarg.optimize.mmse`. It also runs in a separate thread. The optimizer uses `TxTapTuner` objects (`src/pybert/models/tx_tap.py`) to define tap constraints.

### GUI Layer (`src/pybert/gui/`)

- `view.py` — `traits_view`: the complete TraitsUI `View` definition (tabs, controls, plots)
- `handler.py` — `MyHandler(Handler)`: button click handlers; spawns `RunSimThread` / `OptThread`
- `plot.py` — `make_plots()`: creates all Chaco plot containers attached to `PyBERT.plotdata`
- `help.py` — inline help text string

### Utility Modules (`src/pybert/utility/`)

The `utility/` package is a flat re-export via `__init__.py` (star-imports from all submodules). Key modules:
- `channel.py` — `calc_gamma()`, `calc_G()`: physical channel propagation model (Howard Johnson's metallic transmission model)
- `sparam.py` — S-parameter manipulation: `sdd_21()`, `import_channel()`, `cap_mag()`
- `sigproc.py` — signal processing: `make_ctle()`, `calc_eye()`, `find_crossings()`, `resize_zero_pad()`
- `jitter.py` — `calc_jitter()`: jitter decomposition into ISI/DCD/PJ/RJ
- `ibisami.py` — `run_ami_model()`: IBIS-AMI DLL/SO interface
- `math.py` — numerical helpers: `all_combs()`, `safe_log10()`
- `functional.py` — `fst()`, `snd()`, `add_ffe_dfe()`, `get_dfe_weights()`

### Configuration and Results (`src/pybert/configuration.py`, `src/pybert/results.py`)

- Config saved/loaded as YAML (`.yaml`/`.yml`) or legacy pickle (`.pybert_cfg`)
- Results saved/loaded as pickle (`.pybert_data`) — stores plot data arrays
- `PyBertCfg` captures the simulation parameters; `PyBertData` captures the output waveforms and S-parameters

### Parsers (`src/pybert/parsers/`)

- `hspice.py` — CSDF (Common Simulation Data Format) waveform file parser, using the `parsec` PEG library

### Type System (`src/pybert/common.py`)

Type aliases used throughout: `Rvec` (real NDArray), `Cvec` (complex NDArray), `Rmat`, `Cmat`. These are numpy-typed arrays based on `numpy.typing`.

## Testing

Tests use `pytest` with `scope="module"` fixtures that create `PyBERT(run_simulation=False, gui=False)` instances and call `.simulate(initial_run=True)`. The `conftest.py` fixtures to know:

- `dut` — default PyBERT, simulation already run
- `pdut` — parameterized via dict of PyBERT attributes
- `cdut` — YAML-file configured
- `optimization_triplet` — `(PyBERT, MyHandler, DummyInfo)` for testing optimizer
- `ibisami_rx_*` — fixtures using IBIS-AMI models from `models/ibisami/`

## Constraints

- Python `>=3.10,<3.13` (strictly — numpy 1.26.4 does not support 3.13)
- numpy is pinned to `1.26.4` (Chaco compatibility requirement)
- `uv` is the package manager; `pip` is not used directly
- Tox is no longer used (removed as of February 2026)
