"""Simulation results data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   9 May 2017

This Python script provides a data structure for encapsulating the
simulation results data of a PyBERT instance. It was first
created, as a way to facilitate easier pickling, so that a particular
result could be saved and later restored, as a reference waveform.

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""
import pickle
from pathlib import Path
from typing import Union

from chaco.api import ArrayPlotData

RESULTS_FILEDIALOG_WILDCARD = "*.pybert_data"
"""This sets the supported file types in the GUI's save-as or loading dialog."""


class PyBertData:
    """PyBERT simulation results data encapsulation class.

    This class is used to encapsulate that subset of the results data
    for a PyBERT instance, which is to be saved when the user clicks the
    "Save Results" button.
    """

    _item_names = [
        "chnl_h",
        "tx_out_h",
        "ctle_out_h",
        "dfe_out_h",
        "chnl_s",
        "tx_s",
        "ctle_s",
        "dfe_s",
        "tx_out_s",
        "ctle_out_s",
        "dfe_out_s",
        "chnl_p",
        "tx_out_p",
        "ctle_out_p",
        "dfe_out_p",
        "chnl_H",
        "tx_H",
        "ctle_H",
        "dfe_H",
        "tx_out_H",
        "ctle_out_H",
        "dfe_out_H",
        "tx_out",
    ]

    def __init__(self, the_PyBERT, date_created: str, version: str):
        """Copy just that subset of the supplied PyBERT instance's 'plotdata'
        attribute, which should be saved during pickling."""

        plotdata = the_PyBERT.plotdata

        the_data = ArrayPlotData()

        for item_name in self._item_names:
            the_data.set_data(item_name, plotdata.get_data(item_name))

        self.the_data = the_data

        # Generic Information
        self.date_created = date_created
        self.version = version

    def save(self, filepath: Path):
        """Save all of the plot data out to a file.

        Args:
            filepath: The full filepath including the extension to save too.
        """
        with open(filepath, "wb") as the_file:
            pickle.dump(self, the_file)

    # pylint: disable=too-many-branches
    @staticmethod
    def load_from_file(filepath: Union[str, Path], pybert):
        """Recall all the results from a file and load them as reference plots.

        Confirms that the file actually exists and attempts to load back the
        graphs as reference plots in pybert.

        Args:
            filepath: The full filepath including the extension to save too.
            pybert: instance of the main app
        """
        filepath = Path(filepath)  # incase a string was passed convert to a path.

        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        # Right now the loads deserialize back into a `PyBertData` class.
        with open(filepath, "rb") as the_file:
            user_results = pickle.load(the_file)
        if not isinstance(user_results, PyBertData):
            raise ValueError("The data structure read in is NOT of type: ArrayPlotData!")

        # Load the reference plots.
        for prop, value in user_results.the_data.arrays.items():
            pybert.plotdata.set_data(prop + "_ref", value)

        # Add reference plots, if necessary.
        # - time domain
        for container, suffix, has_both in [
            (pybert.plots_h.component_grid.flat, "h", False),
            (pybert.plots_s.component_grid.flat, "s", True),
            (pybert.plots_p.component_grid.flat, "p", False),
        ]:
            if "Reference" not in container[0].plots:
                (ix, prefix) = (0, "chnl")
                item_name = prefix + "_" + suffix + "_ref"
                container[ix].plot(("t_ns_chnl", item_name), type="line", color="darkcyan", name="Inc_ref")
                for ix, prefix in [(1, "tx"), (2, "ctle"), (3, "dfe")]:
                    item_name = prefix + "_out_" + suffix + "_ref"
                    container[ix].plot(("t_ns_chnl", item_name), type="line", color="darkmagenta", name="Cum_ref")
                if has_both:
                    for ix, prefix in [(1, "tx"), (2, "ctle"), (3, "dfe")]:
                        item_name = prefix + "_" + suffix + "_ref"
                        container[ix].plot(("t_ns_chnl", item_name), type="line", color="darkcyan", name="Inc_ref")

        # - frequency domain
        for container, suffix, has_both in [(pybert.plots_H.component_grid.flat, "H", True)]:
            if "Reference" not in container[0].plots:
                (ix, prefix) = (0, "chnl")
                item_name = prefix + "_" + suffix + "_ref"
                container[ix].plot(
                    ("f_GHz", item_name), type="line", color="darkcyan", name="Inc_ref", index_scale="log"
                )
                for ix, prefix in [(1, "tx"), (2, "ctle"), (3, "dfe")]:
                    item_name = prefix + "_out_" + suffix + "_ref"
                    container[ix].plot(
                        ("f_GHz", item_name),
                        type="line",
                        color="darkmagenta",
                        name="Cum_ref",
                        index_scale="log",
                    )
                if has_both:
                    for ix, prefix in [(1, "tx"), (2, "ctle"), (3, "dfe")]:
                        item_name = prefix + "_" + suffix + "_ref"
                        container[ix].plot(
                            ("f_GHz", item_name),
                            type="line",
                            color="darkcyan",
                            name="Inc_ref",
                            index_scale="log",
                        )
