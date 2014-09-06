# Default view definition for PyBERT class.
#
# Original author: David Banas <capn.freako@gmail.com>
# Original date:   August 24, 2014 (Copied from `pybert.py', as part of a major code cleanup.)
#
# Copyright (c) 2014 David Banas; all rights reserved World wide.

from traitsui.api            import View, Item, Group, VGroup, HGroup, Action, Handler, DefaultOverride
from enable.component_editor import ComponentEditor
import time

from pybert_cntrl import *

class MyHandler(Handler):
    """This handler is instantiated by the View and handles user button clicks."""

    def do_get_results(self, info):
        info.object.status = 'Calculating results...'
        update_results(info.object)
        info.object.status = 'Ready.'

    def do_run_dfe(self, info):
        info.object.status = 'Running DFE...'
        my_run_dfe(info.object)
        info.object.status = 'Ready.'

    def do_run_channel(self, info):
        info.object.status = 'Running channel...'
        my_run_channel(info.object)
        info.object.status = 'Ready.'

run_channel = Action(name="RunChannel", action="do_run_channel")
run_dfe     = Action(name="RunDFE",     action="do_run_dfe")
get_results = Action(name="GetResults", action="do_get_results")
    
# Main window layout definition.
traits_view = View(
        VGroup(
            HGroup(
                VGroup(
                    Item(name='ui', label='UI (ps)', show_label=True, enabled_when='True'), #editor=DefaultOverride(mode='spinner'), width=0.5, style='readonly', format_str="%+06.3f"
                    Item(name='nbits',  label='Nbits',          show_label=True, enabled_when='True'),
                    Item(name='nspb',   label='Nspb',    show_label=True, enabled_when='True'),
                    label='Simulation Control', show_border=True,
                ),
                VGroup(
                    Item(name='fc',     label='fc (GHz)',   show_label=True, enabled_when='True'),
                    Item(name='rj',     label='Rj (ps)', show_label=True, enabled_when='True'),
                    Item(name='sj_mag', label='Pj (ps)', ),
                    Item(name='sj_freq', label='f(Pj) (MHz)', ),
                    label='Channel Parameters', show_border=True,
                ),
                VGroup(
                    Item(name='delta_t',      label='Delta-t (ps)',   show_label=True, enabled_when='True'),
                    Item(name='alpha',        label='Alpha',          show_label=True, enabled_when='True'),
                    Item(name='n_lock_ave',   label='Lock Nave.',          show_label=True, enabled_when='True'),
                    Item(name='rel_lock_tol', label='Lock Tol.',          show_label=True, enabled_when='True'),
                    Item(name='lock_sustain', label='Lock Sus.',          show_label=True, enabled_when='True'),
                    label='CDR Parameters', show_border=True,
                ),
                VGroup(
                    Item(name='gain',   label='Gain'),
                    Item(name='n_taps', label='Taps'),
                    Item(name='decision_scaler', label='Level'),
                    Item(name='n_ave', label='Nave.'),
                    label='DFE Parameters', show_border=True,
                ),
                VGroup(
                    Item(name='eye_bits', label='Bits'),
                    label='Results Control', show_border=True,
                ),
            ),
            Group(
                Group(
                    Item('plot_in', editor=ComponentEditor(), show_label=False,),
                    label = 'Channel', id = 'channel'
                ),
                Group(
                    Item('plot_dfe', editor=ComponentEditor(), show_label=False,),
                    label = 'DFE', id = 'dfe'
                ),
                Group(
                    Item('plot_eye', editor=ComponentEditor(), show_label=False,),
                    label = 'Results', id = 'results'
                ),
                Group(
                    Item('ident', style='readonly', show_label=False),
                    label = 'About'
                ),
                layout = 'tabbed', springy = True, id = 'plots',
            ),
            id = 'frame',
        ),
    resizable = True,
    handler = MyHandler(),
    buttons = [run_channel, run_dfe, get_results, "OK"],
    statusbar = "status_str",
    title='PyBERT',
    width=1200, height=800
)

