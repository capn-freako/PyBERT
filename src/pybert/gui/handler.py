"""Action Handler for the traitsui view of the PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
import sys
import webbrowser
from pathlib import Path

from enable.component_editor import ComponentEditor
from pyface.api import OK, FileDialog, MessageDialog
from pyface.image_resource import ImageResource
from traits.api import Instance
from traitsui.api import (  # CloseAction,
    Action,
    CheckListEditor,
    FileEditor,
    Group,
    Handler,
    HGroup,
    Item,
    Menu,
    MenuBar,
    NoButtons,
    ObjectColumn,
    Separator,
    TableEditor,
    TextEditor,
    VGroup,
    View,
    spring,
)

from pybert import __authors__, __copy__, __date__, __version__
from pybert.configuration import CONFIG_LOAD_WILDCARD, CONFIG_SAVE_WILDCARD
from pybert.results import RESULTS_FILEDIALOG_WILDCARD
from pybert.threads.sim import RunSimThread


class MyHandler(Handler):
    """This handler is instantiated by the View and handles user button
    clicks."""

    run_sim_thread = Instance(RunSimThread)

    def do_run_simulation(self, info):
        """Spawn a simulation thread and run with the current settings."""
        the_pybert = info.object
        if self.run_sim_thread and self.run_sim_thread.is_alive():
            pass
        else:
            self.run_sim_thread = RunSimThread()
            self.run_sim_thread.the_pybert = the_pybert
            self.run_sim_thread.start()

    def do_stop_simulation(self):
        """Kill the simulation thread."""
        if self.run_sim_thread and self.run_sim_thread.is_alive():
            self.run_sim_thread.stop()

    def do_save_cfg(self, info):
        """Save the configuration.

        If no config file is set, use the `self.do_save_cfg_as()` method to prompt the user.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        configuration_file = Path(pybert.cfg_file)
        if pybert.cfg_file and configuration_file.exists():
            pybert.save_configuration(configuration_file)
        else:  # A configuration file hasn't been set yet so use the save-as method
            self.do_save_cfg_as(info)

    def do_save_cfg_as(self, info):
        """Prompt the user to choose where to save the config and save it.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        dialog = FileDialog(
            action="save as",
            wildcard=CONFIG_SAVE_WILDCARD,
            default_path=pybert.cfg_file,
        )
        if dialog.open() == OK:
            pybert.save_configuration(Path(dialog.path))

    def do_load_cfg(self, info):
        """Prompt the user to choose where to load the config from and load it.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        dialog = FileDialog(
            action="open",
            wildcard=CONFIG_LOAD_WILDCARD,
            default_path=pybert.cfg_file,
        )
        if dialog.open() == OK:
            pybert.load_configuration(Path(dialog.path))

    def do_save_data(self, info):
        """Prompt the user to choose where to save the results and save it.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        dialog = FileDialog(action="save as", wildcard=RESULTS_FILEDIALOG_WILDCARD, default_path=pybert.data_file)
        if dialog.open() == OK:
            pybert.save_results(Path(dialog.path))

    def do_load_data(self, info):
        """Prompt the user to choose where to load the results from and load it.

        Pybert will load these as "reference" plots for the responses.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        dialog = FileDialog(action="open", wildcard=RESULTS_FILEDIALOG_WILDCARD, default_path=pybert.data_file)
        if dialog.open() == OK:
            pybert.load_results(Path(dialog.path))

    def do_clear_data(self, info):
        """If any reference or prior results has been loaded, clear it.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        pybert.clear_reference_from_plots()

    def toggle_debug_clicked(self, info):
        """Toggle whether debug mode is enabled or not."""
        # pylint: disable=no-self-use
        info.object.debug = not info.object.debug

    def getting_started_clicked(self, info):
        """Open up Pybert's wiki."""
        # pylint: disable=no-self-use,unused-argument
        webbrowser.open("https://github.com/capn-freako/PyBERT/wiki")

    def show_about_clicked(self, info):
        """Open a dialog and show the user the about info."""
        # pylint: disable=no-self-use,unused-argument
        dialog = MessageDialog(
            title="About Pybert",
            message=f"PyBERT v{__version__} - a serial communication link design tool, written in Python.",
            informative=(
                f"{__authors__}<BR>\n" f"{__date__}<BR><BR>\n\n" f"{__copy__};<BR>\n" "All rights reserved World wide."
            ),
            severity="information",
        )
        dialog.open()

    def close_app(self, info):
        """Close the app.

        Can't use CloseAction until https://github.com/enthought/traitsui/issues/1442 is resolved.
        """
        # pylint: disable=no-self-use,unused-argument
        sys.exit(0)
