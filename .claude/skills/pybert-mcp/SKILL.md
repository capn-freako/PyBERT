---
name: pybert-mcp
description: Drive PyBERT's MCP server to run headless serial-link BER simulations, read/write configuration, browse saved results, and inspect IBIS-AMI models, without opening the GUI. Use when asked to run a PyBERT simulation, tune Tx/Rx/channel/CTLE/DFE settings, evaluate link performance (BER, jitter), compare configurations, or introspect an IBIS/AMI file.
allowed-tools: mcp__pybert__get_default_config mcp__pybert__get_config mcp__pybert__set_config mcp__pybert__run_simulation mcp__pybert__inspect_results_file mcp__pybert__list_ibis_models mcp__pybert__inspect_ibis_model
---

# PyBERT MCP

Seven tools, registered as `mcp__pybert__*` (server defined in `.mcp.json`, implementation in
`src/pybert/mcp/server.py`). Every call constructs a fresh, isolated `PyBERT` instance
server-side — there is no session state between calls, and no shared cache. Whatever settings you
want in effect for a given `run_simulation` (or `set_config`) call must be present in the `config`
dict you pass to *that* call.

## Prerequisites

- The server needs the `mcp` extra installed: `uv sync --extra mcp`. If it isn't, tool calls will
  fail at the transport level (the server process itself exits with a clear "MCP support requires:
  pip install 'pipbert[mcp]'" message if invoked directly, but from inside a session that just
  looks like the server failing to start — tell the user to run `uv sync --extra mcp` if that
  happens).
- **File paths are resolved relative to the server process's working directory**, which is this
  repo's root (set by `.mcp.json`) — not the user's shell cwd, not this skill's directory. Always
  pass repo-relative paths (e.g. `"models/ibisami/example_rx.ibs"`), matching how paths are used
  throughout this repo's own tests and examples (`tests/test_mcp.py`, `examples/mcp_client.py`).

## 1. Configuration — `get_default_config` / `get_config` / `set_config`

- Always start from a *full* config dict — `get_default_config()` or `get_config(config_file)` —
  never hand-build one. It has ~88 keys; real names include `bit_rate`, `nbits`, `eye_bits`,
  `tx_taps`, `rx_bw`, `peak_mag`, `rx_use_viterbi`, `use_dfe`. Don't guess a key name; read it off
  the dict you just fetched.
- Both `set_config` and `run_simulation` validate keys strictly: an unrecognized key (e.g. a typo
  like `"bit_rat"`) raises an error naming the bad key(s). That's intentional — treat it as "you
  used a wrong key name," not as a bug to route around.
- `set_config(overrides, out_file, base_config_file=None)` layers `overrides` onto
  `base_config_file` (or the defaults) and writes the result to `out_file` as YAML — it does not
  run a simulation. Use it to persist a tuned configuration for later use with the GUI
  (`pybert -c out_file`) or CLI (`pybert sim out_file`).

## 2. Running a simulation — `run_simulation`

`run_simulation(config, results_file=None)` runs headless and returns scalar metrics directly:
`status`, `total_perf`, `jitter_perf`, `n_errs_dfe`, `n_errs_viterbi`, `chnl_dly`, `isi_dfe`,
`dcd_dfe`, `pj_dfe`, `rj_dfe`, `perf_info` (an HTML table), `jitter_info`.

- **Checking success**: `status` is a free-text log message, not a boolean — don't treat any
  non-empty string as success. On success it ends as `"Ready."`. On failure it starts with
  `"Exception: ..."` or reads `"Aborted Simulation"`. Check for those.
- `n_errs_viterbi == -1` means Viterbi/FEC decoding wasn't enabled for that run (`rx_use_viterbi`
  was false) — it's a sentinel, not an error.
- For tuning/comparison loops, shrink `nbits`/`eye_bits` first (e.g. `1000`/`500`) — full-length
  runs are much slower and unnecessary until you've converged on settings worth a final,
  full-length confirmation run.
- Only pass `results_file` when you actually need `inspect_results_file` afterward — it triggers
  extra plot-data population and file I/O a metrics-only call doesn't need.

## 3. Browsing saved results — `inspect_results_file`

Only works on files written by `run_simulation(..., results_file=...)` (or `pybert sim`). It
reports `{array_name: {shape, dtype, min, max, mean}}` per waveform (impulse/step/pulse responses,
etc.) — **never** raw samples, and **never** scalar performance metrics (those exist only live,
via `run_simulation`'s return value — they aren't persisted in `.pybert_data` files). If you need
actual sample values, there is no tool for that here; say so rather than inventing numbers.

## 4. IBIS-AMI introspection — `list_ibis_models` / `inspect_ibis_model`

Two-step, always in this order:
1. `list_ibis_models(ibis_file)` — confirms the file parses (`parsing_errors == "Success!"`) and
   lists valid `components`/`models` names.
2. `inspect_ibis_model(ibis_file, model_name)` — pass an exact name from step 1's `models` list.

`inspect_ibis_model` returns `null` for `zout`/`slew` on Rx-only models and for `zin` on Tx-only
models — that's expected (those are driver-only / receiver-only properties), not missing data.
`ami_files` lists DLL/SO + `.ami` paths per platform/bitness; this tool never executes the AMI
model itself.

## Common workflows

**Tune a parameter and compare performance:**
1. `get_default_config()` (or `get_config` on an existing file) → `cfg`.
2. Shrink `cfg["nbits"]` / `cfg["eye_bits"]` for speed.
3. For each candidate setting: mutate the relevant key(s) on a copy of `cfg` (e.g. `cfg["rx_bw"]`,
   `cfg["peak_mag"]`, `cfg["tx_taps"]`) → `run_simulation(cfg)` → compare `total_perf` /
   `jitter_perf` across variants, checking `status` on each.
4. Once satisfied, run once more at full-length `nbits`/`eye_bits` for the number you actually
   report, optionally with `results_file` set. Report the numbers a tool call actually returned —
   never estimate or interpolate a result.

**Persist a tuned configuration:** `set_config(overrides, out_file)` once you've found values
worth keeping.

## Pitfalls

- Don't fabricate simulation numbers or waveform contents. If a call errors or a result looks odd,
  report it as-is rather than smoothing it over.
- Config-key validation errors are the server telling you the key doesn't exist — fix the key name
  using the dict from `get_default_config`/`get_config`, don't retry the same key.
