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
        HGroup(
            VGroup(
                Item(name='ui',          label='UI (ps)',  tooltip="unit interval", show_label=True, enabled_when='True'),
                #editor=DefaultOverride(mode='spinner'), width=0.5, style='readonly', format_str="%+06.3f"
                Item(name='nbits',       label='Nbits',    tooltip="# of bits to run", ),
                Item(name='nspb',        label='Nspb',     tooltip="# of samples per bit", ),
                Item(name='pattern_len', label='PatLen',   tooltip="length of random pattern to use to construct bit stream", ),
                Item(name='eye_bits',    label='EyeBits',  tooltip="# of bits to use to form eye diagrams", ),
                label='Simulation Control', show_border=True,
            ),
            VGroup(
                Item(name='Theta0',  label='Loss Tan.',   tooltip="dielectric loss tangent", ),
                Item(name='Z0',      label='Z0 (Ohms)',   tooltip="characteristic differential impedance", ),
                Item(name='v0',      label='v_rel (c)',   tooltip="normalized propagation velocity", ),
                Item(name='l_ch',    label='Length (m)',  tooltip="interconnect length", ),
                Item(name='rn',      label='Rn (V)',      tooltip="standard deviation of random noise", ),
                label='Channel Parameters', show_border=True,
            ),
            VGroup(
                Item(name='vod',     label='Vod (V)',      tooltip="Tx output voltage into matched load", ),
                Item(name='rs',      label='Rs (Ohms)',    tooltip="Tx differential source impedance", ),
                Item(name='cout',    label='Cout (pF)',    tooltip="Tx parasitic output capacitance (each pin)", ),
                Item(name='pn_mag',  label='Pn (V)',       tooltip="peak magnitude of periodic noise", ),
                Item(name='pn_freq', label='f(Pn) (MHz)',  tooltip="frequency of periodic noise", ),
                label='Tx Analog', show_border=True,
            ),
            VGroup(
                Item(name='pretap',  label='Pre-tap',      tooltip="pre-cursor tap weight", ),
                Item(name='posttap', label='Post-tap',     tooltip="post-cursor tap weight", ),
                label='Tx Equalization', show_border=True,
            ),
            VGroup(
                Item(name='rin',     label='Rin (Ohms)',  tooltip="Rx differential input impedance", ),
                Item(name='cin',     label='Cin (pF)',    tooltip="Rx parasitic input capacitance (each pin)", ),
                Item(name='cac',     label='Cac (uF)',    tooltip="Rx a.c. coupling capacitance (each pin)", ),
                Item(name='rx_bw',     label='Bandwidth (GHz)',      tooltip="unequalized signal path bandwidth (GHz).", ),
                label='Rx Analog', show_border=True,
            ),
            VGroup(
                Item(name='peak_freq', label='CTLE fp (GHz)',        tooltip="CTLE peaking frequency (GHz)", ),
                Item(name='peak_mag',  label='CTLE boost (dB)',      tooltip="CTLE peaking magnitude (dB)", ),
                Item(name='use_dfe',   label='Use DFE',              tooltip="Include DFE in simulation.", ),
                Item(name='sum_ideal', label='Ideal DFE',            tooltip="Use ideal DFE. (performance boost)", ),
                label='Rx Equalization', show_border=True,
            ),
            VGroup(
                Item(name='delta_t',      label='Delta-t (ps)',  tooltip="magnitude of CDR proportional branch", ),
                Item(name='alpha',        label='Alpha',         tooltip="relative magnitude of CDR integral branch", ),
                Item(name='n_lock_ave',   label='Lock Nave.',    tooltip="# of UI estimates to average, when determining lock", ),
                Item(name='rel_lock_tol', label='Lock Tol.',     tooltip="relative tolerance for determining lock", ),
                Item(name='lock_sustain', label='Lock Sus.',     tooltip="length of lock determining hysteresis vector", ),
                label='CDR Parameters', show_border=True,
            ),
            VGroup(
                Item(name='gain',            label='Gain',   tooltip="error feedback gain", ),
                Item(name='n_taps',          label='Taps',   tooltip="# of taps", ),
                Item(name='decision_scaler', label='Level',  tooltip="target output magnitude", ),
                Item(name='n_ave',           label='Nave.',  tooltip="# of CDR adaptations per DFE adaptation", ),
                Item(name='sum_bw',    label='BW (GHz)', tooltip="summing node bandwidth", ),
                label='DFE Parameters', show_border=True,
            ),
            VGroup(
                Item(name='thresh',          label='Pj Thresh.',   tooltip="Threshold for identifying periodic jitter spectral elements. (sigma)", ),
                label='Analysis Parameters', show_border=True,
            ),
            label = 'Config.', id = 'config',
            layout = 'flow',
        ),
        Group(
            Item('plots_dfe', editor=ComponentEditor(), show_label=False,),
            label = 'DFE', id = 'plots_dfe'
        ),
        Group(
            Item('plots_h', editor=ComponentEditor(), show_label=False,),
            label = 'Impulse Responses', id = 'plots_h'
        ),
        Group(
            Item('plots_s', editor=ComponentEditor(), show_label=False,),
            label = 'Step Responses', id = 'plots_s'
        ),
        Group(
            Item('plots_H', editor=ComponentEditor(), show_label=False,),
            label = 'Frequency Responses', id = 'plots_H'
        ),
        Group(
            Item('plots_out', editor=ComponentEditor(), show_label=False,),
            label = 'Outputs', id = 'plots_out'
        ),
        Group(
            Item('plots_eye', editor=ComponentEditor(), show_label=False,),
            label = 'Eye Diagrams', id = 'plots_eye'
        ),
        Group(
            Item('plots_jitter_dist', editor=ComponentEditor(), show_label=False,),
            label = 'Jitter Distributions', id = 'plots_jitter_dist'
        ),
        Group(
            Item('plots_jitter_spec', editor=ComponentEditor(), show_label=False,),
            label = 'Jitter Spectrums', id = 'plots_jitter_spec'
        ),
        Group(
            Item('plots_bathtub', editor=ComponentEditor(), show_label=False,),
            label = 'Bathtub Curves', id = 'plots_bathtub'
        ),
        Group(
            Item('jitter_info', style='readonly', show_label=False),
            label = 'Jitter Info'
        ),
        Group(
            Item('ident', style='readonly', show_label=False),
            Item('perf_info', style='readonly', show_label=False),
            label = 'About'
        ),
        Group(
            Item(label='Instructions', springy=True, ),
            label = 'Help'
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

