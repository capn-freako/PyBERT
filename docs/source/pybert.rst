====================
*PyBERT* Main Module
====================

Main Class
**********

.. autoclass:: pybert.pybert.PyBERT
   :members: __init__

Custom Threads
**************

A separate thread is used for optimization, in order to preserve GUI responsiveness to user input while optimizing.
And all custom threads used in PyBERT are derived from `StoppableThread`, so that any optimization may be aborted by the user.

.. autoclass:: pybert.threads.stoppable.StoppableThread
   :members:

.. autoclass:: pybert.threads.optimization.OptThread
   :members:

Tx FIR Tap Tuner
****************

.. autoclass:: pybert.pybert.TxTapTuner
   :members:
