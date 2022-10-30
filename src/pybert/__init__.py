"""A package of Python modules, used by the *PyBERT* application.

.. moduleauthor:: David Banas <capn.freako@gmail.com>

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by:      Mark Marlett <mark.marlett@gmail.com>

The application source is divided among several files, as follows:

    pybert.py       - This file. The M in MVC, it contains:
                      - independent variable declarations
                      - default initialization
                      - the definitions of those dependent variables, which are handled
                        automatically by the Traits/UI machinery.

    view.py  - The V in MVC, it contains the main window layout definition, as
                      well as the definitions of user invoked actions
                      (i.e.- buttons).

    simulation.py - The C in MVC, it contains the definitions for those dependent
                      variables, which are updated not automatically by
                      the Traits/UI machinery, but rather by explicit
                      user action (i.e. - button clicks).

    help.py  - Contents for the "Help" tab of the GUI.

    plot.py  - Contains all plot definitions.

    utility.py  - Contains general purpose utility functionality.

    configuration.py   - Defines the data structure for storing PyBERT
                      configurations, so they may be saved and later
                      restored.

    results.py  - Defines the data structure for storing PyBERT
                      simulation results, so they may be saved and later
                      recalled as reference waveforms for comparison.

    dfe.py          - Contains the decision feedback equalizer model.

    cdr.py          - Contains the clock data recovery unit model.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""
__version__ = "3.5.8"
__date__ = "October 5, 2022"
__authors__ = "David Banas & David Patterson"
__copy__ = "Copyright (c) 2014 David Banas, 2019 David Patterson"
