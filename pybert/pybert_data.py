"""
Simulation results data encapsulation, for PyBERT.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   9 May 2017

This Python script provides a data structure for encapsulating the
simulation results data of a PyBERT instance. It was first
created, as a way to facilitate easier pickling, so that a particular
result could be saved and later restored, as a reference waveform.

Copyright (c) 2017 by David Banas; All rights reserved World wide.
"""
from chaco.api import ArrayPlotData


class PyBertData:
    """
    PyBERT simulation results data encapsulation class.

    This class is used to encapsulate that subset of the results
    data for a PyBERT instance, which is to be saved when the user
    clicks the "Save Results" button.
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
        "rx_in",
    ]

    def __init__(self, the_PyBERT):
        """
        Copy just that subset of the supplied PyBERT instance's
        'plotdata' attribute, which should be saved during pickling.
        """

        plotdata = the_PyBERT.plotdata

        the_data = ArrayPlotData()

        for item_name in self._item_names:
            the_data.set_data(item_name, plotdata.get_data(item_name))

        self.the_data = the_data
