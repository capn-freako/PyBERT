from _typeshed import Incomplete
from pathlib import Path
from typing import Any

NOTEBOOK: Incomplete

def run_notebook(ibis_file: Path, notebook: Path, out_dir: Path | None = None, notebook_params: dict[str, Any] | None = None) -> None: ...
def main(notebook, out_dir, params, ibis_file, bit_rate, debug, is_tx, nspui, nbits, plot_t_max, f_max, f_step, fig_x, fig_y) -> None: ...
