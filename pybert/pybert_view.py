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

    def do_run_simulation(self, info):
        info.object.status = 'Running channel...'
        my_run_simulation(info.object)
        info.object.status = 'Ready.'

run_simulation = Action(name="Run", action="do_run_simulation")
    
# Main window layout definition.
traits_view = View(
    Group(
        VGroup(
            HGroup(
                VGroup(
                    Item(name='ui',          label='UI (ps)', width=-75,  tooltip="unit interval", show_label=True, enabled_when='True'),
                    #editor=DefaultOverride(mode='spinner'), width=0.5, style='readonly', format_str="%+06.3f"
                    Item(name='nbits',       label='Nbits',   width=-75,  tooltip="# of bits to run", ),
                    Item(name='nspb',        label='Nspb',    width=-75,  tooltip="# of samples per bit", ),
                    Item(name='pattern_len', label='PatLen',  width=-75,  tooltip="length of random pattern to use to construct bit stream", ),
                    Item(name='eye_bits',    label='EyeBits', width=-75,  tooltip="# of bits to use to form eye diagrams", ),
                    label='Simulation Control', show_border=True,
                ),
                VGroup(
                    Item(name='vod',     label='Vod (V)',     width=-75,  tooltip="Tx output voltage into matched load", ),
                    Item(name='rs',      label='Rs (Ohms)',   width=-75,  tooltip="Tx differential source impedance", ),
                    Item(name='cout',    label='Cout (pF)',   width=-75,  tooltip="Tx parasitic output capacitance (each pin)", ),
                    Item(name='pn_mag',  label='Pn (V)',      width=-75,  tooltip="peak magnitude of periodic noise", ),
                    Item(name='pn_freq', label='f(Pn) (MHz)', width=-75,  tooltip="frequency of periodic noise", ),
                    Item(name='pretap',  label='Pre-tap',     width=-75,  tooltip="pre-cursor tap weight", ),
                    Item(name='posttap', label='Post-tap',    width=-75,  tooltip="post-cursor tap weight", ),
                    label='Tx Parameters', show_border=True,
                ),
                VGroup(
                    Item(name='Theta0',  label='Loss Tan.',  width=-75,  tooltip="dielectric loss tangent", ),
                    Item(name='Z0',      label='Z0 (Ohms)',  width=-75,  tooltip="characteristic differential impedance", ),
                    Item(name='v0',      label='v_rel (c)',  width=-75,  tooltip="normalized propagation velocity", ),
                    Item(name='l_ch',    label='Length (m)', width=-75,  tooltip="interconnect length", ),
                    Item(name='rn',      label='Rn (V)',     width=-75,  tooltip="standard deviation of random noise", ),
                    label='Channel Parameters', show_border=True,
                ),
                VGroup(
                    Item(name='rin',     label='Rin (Ohms)', width=-75,  tooltip="Rx differential input impedance", ),
                    Item(name='cin',     label='Cin (pF)',   width=-75,  tooltip="Rx parasitic input capacitance (each pin)", ),
                    Item(name='cac',     label='Cac (uF)',   width=-75,  tooltip="Rx a.c. coupling capacitance (each pin)", ),
                    Item(name='use_dfe',   label='Use DFE',              tooltip="Include DFE in simulation.", ),
                    Item(name='sum_ideal', label='Ideal DFE',            tooltip="Use ideal DFE. (performance boost)", ),
                    label='Rx Parameters', show_border=True,
                ),
                VGroup(
                    Item(name='delta_t',      label='Delta-t (ps)', width=-75,  tooltip="magnitude of CDR proportional branch", ),
                    Item(name='alpha',        label='Alpha',        width=-75,  tooltip="relative magnitude of CDR integral branch", ),
                    Item(name='n_lock_ave',   label='Lock Nave.',   width=-75,  tooltip="# of UI estimates to average, when determining lock", ),
                    Item(name='rel_lock_tol', label='Lock Tol.',    width=-75,  tooltip="relative tolerance for determining lock", ),
                    Item(name='lock_sustain', label='Lock Sus.',    width=-75,  tooltip="length of lock determining hysteresis vector", ),
                    label='CDR Parameters', show_border=True,
                ),
                VGroup(
                    Item(name='gain',            label='Gain',  width=-75,  tooltip="error feedback gain", ),
                    Item(name='n_taps',          label='Taps',  width=-75,  tooltip="# of taps", ),
                    Item(name='decision_scaler', label='Level', width=-75,  tooltip="target output magnitude", ),
                    Item(name='n_ave',           label='Nave.', width=-75,  tooltip="# of CDR adaptations per DFE adaptation", ),
                    Item(name='sum_bw',    label='BW (GHz)', width=-75, tooltip="summing node bandwidth", ),
                    label='DFE Parameters', show_border=True,
                ),
            ),
            Item(label='Instructions', springy=True, ),
            label = 'Config.', id = 'config'
        ),
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
            label = 'Eye Diagrams', id = 'results'
        ),
        Group(
            Item('plot_jitter', editor=ComponentEditor(), show_label=False,),
            label = 'Jitter', id = 'jitter'
        ),
        Group(
            Item('jitter_info', style='readonly', show_label=False),
            label = 'Jitter Info'
        ),
        Group(
            Item('ident', style='readonly', show_label=False),
            label = 'About'
        ),
        layout = 'tabbed', springy = True, id = 'tabs',
    ),
    resizable = True,
    handler = MyHandler(),
    buttons = [run_simulation, "OK"],
    statusbar = "status_str",
    title='PyBERT',
    width=1200, height=800
)

