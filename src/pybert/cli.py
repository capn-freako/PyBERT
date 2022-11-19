"""Main Entry Point for the PyBERT GUI when using the CLI."""
from pathlib import Path

import click

from pybert.gui.view import traits_view
from pybert.pybert import PyBERT


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
@click.version_option()
@click.option("--config", "-c", type=click.Path(exists=True), help="Load an existing configuration.")
@click.option("--results", "-r", type=click.Path(exists=True), help="Load results from a prior run.")
def cli(ctx, config, results):
    """Serial communication link bit error rate tester."""

    if ctx.invoked_subcommand is None:
        pybert = PyBERT()

        if config:
            pybert.load_configuration(config)
        if results:
            pybert.load_results(results)

        # Show the GUI.
        pybert.configure_traits(view=traits_view)


@cli.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("config", type=click.Path(exists=True))
def run(config):
    """Run a simulation without opening the GUI."""
    pybert = PyBERT(run_simulation=False, gui=False)
    pybert.load_configuration(config)
    pybert.simulate(initial_run=True, update_plots=True)
    pybert.save_results(Path(config).with_suffix(".pybert_data"))
