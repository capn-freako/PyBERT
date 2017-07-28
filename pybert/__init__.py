"""
A package of Python modules, used by the *PyBERT* application.

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
                
    pybert_view.py  - The V in MVC, it contains the main window layout definition, as
                      well as the definitions of user invoked actions
                      (i.e.- buttons).

    pybert_cntrl.py - The C in MVC, it contains the definitions for those dependent
                      variables, which are updated not automatically by
                      the Traits/UI machinery, but rather by explicit
                      user action (i.e. - button clicks).

    pybert_help.py  - Contents for the "Help" tab of the GUI.

    pybert_plot.py  - Contains all plot definitions.

    pybert_util.py  - Contains general purpose utility functionality.

    dfe.py          - Contains the decision feedback equalizer model.

    cdr.py          - Contains the clock data recovery unit model.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

# import pybert
# import pybert_view
# import pybert_cntrl
# import pybert_util
# import dfe
# import cdr
# 
# __all__ = ['pybert', 'pybert_view', 'pybert_cntrl', 'pybert_util', 'dfe', 'cdr']

__version__ = '2.2.0'

