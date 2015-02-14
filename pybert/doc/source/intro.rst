.. Modules in the 'pybert' package

General description of *pybert* package source code structure
-------------------------------------------------------------

The Python source is divided among several modules, as follows:

    pybert.py       - The "model" component. It contains:
                      - independent variable declarations
                      - default initialization
                      - the definitions of those dependent variables, which are handled
                        automatically by the Traits/UI machinery.
                
    pybert_view.py  - The "view" component. It contains the main window layout definition, as
                      well as the definitions of user invoked actions
                      (i.e.- buttons).

    pybert_cntrl.py - The "controller" component. It contains the definitions for those dependent
                      variables, which are updated not automatically by
                      the Traits/UI machinery, but rather by explicit
                      user action (i.e. - button clicks).

    pybert_util.py  - Contains general purpose utility functionality.

    dfe.py          - Contains the decision feedback equalizer (DFE) model.

    cdr.py          - Contains the clock data recovery (CDR) unit model.

