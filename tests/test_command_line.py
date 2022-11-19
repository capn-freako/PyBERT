"""Unit test coverage to make sure that the added cli commands function like the gui."""
import logging

from click.testing import CliRunner

from pybert import __version__
from pybert.cli import cli
from pybert.pybert import PyBERT


def test_cli_version():
    """Make sure that the command line interface functions enough to print a version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_sim(caplog, tmp_path):
    """Make sure that pybert can run without a gui and generate a results file output."""
    caplog.set_level(logging.DEBUG)

    app = PyBERT(run_simulation=False, gui=False)
    config_file = tmp_path.joinpath("config.yaml")
    app.save_configuration(config_file)
    assert config_file.exists()

    runner = CliRunner()
    result = runner.invoke(cli, ["sim", str(config_file)])
    saved_results = config_file.with_suffix(".pybert_data")
    assert result.exit_code == 0
    assert saved_results.exists()
