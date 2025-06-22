===========================
Modules in *PyBERT* Package
===========================

Top Level Modules (`pybert`)
****************************

.. toctree::
   :maxdepth: 1

   pybert

pybert.common - Common miscellaneous definitions and aliases.
-------------------------------------------------------------

.. automodule:: pybert.common
   :members: Real,Comp,Rvec,Cvec,Rmat,Cmat
   :member-order: bysource

pybert.configuration - Data structure for saving *PyBERT* configuration.
------------------------------------------------------------------------

.. automodule:: pybert.configuration
   :members:

pybert.results - Data structure for saving *PyBERT* results.
------------------------------------------------------------

.. automodule:: pybert.results
   :members:

pybert.cli - Command line interface.
------------------------------------------------------------

.. automodule:: pybert.cli
   :members:
   
Models (`pybert.models`)
************************

.. automodule:: pybert.models

.. toctree::
   :maxdepth: 1

   bert
   tx_tap
   dfe
   cdr
   viterbi

GUI Elements (`pybert.gui`)
***************************

.. automodule:: pybert.gui

.. toctree::
   :maxdepth: 1

   view
   plot
   handler
   help

Parsers (`pybert.parsers`)
**************************

.. automodule:: pybert.parsers

Threads (`pybert.threads`)
**************************

.. automodule:: pybert.threads

.. toctree::
   :maxdepth: 1

   stoppable
   sim
   optimization

Utility (`pybert.utility`)
**************************

.. automodule:: pybert.utility

.. toctree::
   :maxdepth: 1

   channel
   ibisami
   jitter
   math
   python
   sigproc
   sparam
