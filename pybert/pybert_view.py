"""
Default view definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>
Original date:   August 24, 2014 (Copied from `pybert.py', as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

from threading               import Thread

from traits.api              import Instance
from traitsui.api            import View, Item, Group, VGroup, HGroup, Action, Handler, DefaultOverride, CheckListEditor, StatusItem, TextEditor
from enable.component_editor import ComponentEditor

from pybert_cntrl            import my_run_sweeps

class RunSimThread(Thread):
    'Used to run the simulation in its own thread, in order to preserve GUI responsiveness.'

    def run(self):
        my_run_sweeps(self.the_pybert)

class MyHandler(Handler):
    """This handler is instantiated by the View and handles user button clicks."""

    run_sim_thread = Instance(RunSimThread)

    def do_run_simulation(self, info):
        the_pybert = info.object
        if self.run_sim_thread and self.run_sim_thread.isAlive():
            pass
        else:
            self.run_sim_thread            = RunSimThread()
            self.run_sim_thread.the_pybert = the_pybert
            self.run_sim_thread.start()

run_simulation = Action(name="Run",     action="do_run_simulation")
    
# Main window layout definition.
traits_view = View(
    Group(
        VGroup(
            HGroup(
                HGroup(
                    VGroup(
                        Item(name='bit_rate',    label='Bit Rate (Gbps)',  tooltip="bit rate", show_label=True, enabled_when='True',
                            #editor=DefaultOverride(mode='spinner'), width=0.5, style='readonly', format_str="%+06.3f"
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)
                        ),
                        Item(name='nbits',       label='Nbits',    tooltip="# of bits to run",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=int)
                        ),
                        Item(name='nspb',        label='Nspb',     tooltip="# of samples per bit",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=int)
                        ),
                        Item(name='pattern_len', label='PatLen',   tooltip="length of random pattern to use to construct bit stream",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=int)
                        ),
                        Item(name='eye_bits',    label='EyeBits',  tooltip="# of bits to use to form eye diagrams",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=int)
                        ),
                        Item(name='mod_type',    label='Modulation', tooltip="line signalling/modulation scheme",
                            editor=CheckListEditor(values=[(0, 'NRZ'), (1, 'Duo-binary'), (2, 'PAM-4'),])
                        ),
                    ),
                    VGroup(
                        Item(name='do_sweep',    label='Do Sweep',    tooltip="Run parameter sweeps.", ),
                        Item(name='sweep_aves',  label='SweepAves',   tooltip="# of trials, per sweep, for averaging.", enabled_when='do_sweep == True'),
                        Item(name='vod',         label='Vod (V)',     tooltip="Tx output voltage into matched load", ),
                        Item(name='rn',          label='Rn (V)',      tooltip="standard deviation of random noise", ),
                        Item(name='pn_mag',      label='Pn (V)',      tooltip="peak magnitude of periodic noise", ),
                        Item(name='pn_freq',     label='f(Pn) (MHz)', tooltip="frequency of periodic noise", ),
                    ),
                    label='Simulation Control', show_border=True,
                ),
                HGroup(
                    VGroup(
                        Item(name='use_ch_file', label='fromFile', tooltip='Select channel impulse/step response from file.', ),
                        Item(name='ch_file', label='Filename',    enabled_when='use_ch_file == True'),
                        Item(name='Theta0',  label='Loss Tan.',   enabled_when='use_ch_file == False', tooltip="dielectric loss tangent", ),
                        Item(name='Z0',      label='Z0 (Ohms)',   enabled_when='use_ch_file == False', tooltip="characteristic differential impedance", ),
                        Item(name='v0',      label='v_rel (c)',   enabled_when='use_ch_file == False', tooltip="normalized propagation velocity", ),
                        Item(name='l_ch',    label='Length (m)',  enabled_when='use_ch_file == False', tooltip="interconnect length", ),
                    ),
                    VGroup(
                        Item(name='impulse_length', label='Impl. Len. (ns)', tooltip="Manual impulse response length override", ),
                        Item(name='rs',      label='Tx_Rs (Ohms)',   enabled_when='use_ch_file == False', tooltip="Tx differential source impedance", ),
                        Item(name='cout',    label='Tx_Cout (pF)',   enabled_when='use_ch_file == False', tooltip="Tx parasitic output capacitance (each pin)", ),
                        Item(name='rin',     label='Rx_Rin (Ohms)',  enabled_when='use_ch_file == False', tooltip="Rx differential input impedance", ),
                        Item(name='cin',     label='Rx_Cin (pF)',    enabled_when='use_ch_file == False', tooltip="Rx parasitic input capacitance (each pin)", ),
                        Item(name='cac',     label='Rx_Cac (uF)',    enabled_when='use_ch_file == False', tooltip="Rx a.c. coupling capacitance (each pin)", ),
                    ),
                    label='Channel Parameters', show_border=True,
                ),
            ),
            HGroup(
                VGroup(
                    HGroup(
                        Item(name='pretap_enable', label='Enable',    tooltip="Enable this tap."),
                        Item(name='pretap',       label='Pre-tap',    tooltip="pre-cursor tap weight",         enabled_when='pretap_enable == True' ),
                        Item(name='pretap_sweep', label='SweepTo',    tooltip="Perform automated parameter sweep."),
                        Item(name='pretap_final', show_label=False,   tooltip="final pretap value",            enabled_when='pretap_sweep == True'),
                        Item(name='pretap_steps', label='# of Steps', tooltip="number of pretap steps",        enabled_when='pretap_sweep == True'),
                    ),
                    HGroup(
                        Item(name='posttap_enable', label='Enable',    tooltip="Enable this tap."),
                        Item(name='posttap',       label='Post-tap',   tooltip="post-cursor tap weight",       enabled_when='posttap_enable == True' ),
                        Item(name='posttap_sweep', label='SweepTo',    tooltip="Perform automated parameter sweep."),
                        Item(name='posttap_final', show_label=False,   tooltip="final posttap value",          enabled_when='posttap_sweep == True'),
                        Item(name='posttap_steps', label='# of Steps', tooltip="number of posttap steps",      enabled_when='posttap_sweep == True'),
                    ),
                    HGroup(
                        Item(name='posttap2_enable', label='Enable',    tooltip="Enable this tap."),
                        Item(name='posttap2',       label='Post-tap2',  tooltip="2nd post-cursor tap weight",  enabled_when='posttap2_enable == True' ),
                        Item(name='posttap2_sweep', label='SweepTo',    tooltip="Perform automated parameter sweep."),
                        Item(name='posttap2_final', show_label=False,   tooltip="final value",                 enabled_when='posttap_sweep == True'),
                        Item(name='posttap2_steps', label='# of Steps', tooltip="number of steps",             enabled_when='posttap_sweep == True'),
                    ),
                    HGroup(
                        Item(name='posttap3_enable', label='Enable',    tooltip="Enable this tap."),
                        Item(name='posttap3',       label='Post-tap3',  tooltip="3rd post-cursor tap weight",  enabled_when='posttap3_enable == True' ),
                        Item(name='posttap3_sweep', label='SweepTo',    tooltip="Perform automated parameter sweep."),
                        Item(name='posttap3_final', show_label=False,   tooltip="final value",                 enabled_when='posttap_sweep == True'),
                        Item(name='posttap3_steps', label='# of Steps', tooltip="number of steps",             enabled_when='posttap_sweep == True'),
                    ),
                    label='Tx Equalization', show_border=True,
                ),
                HGroup(
                    VGroup(
                        Item(name='peak_freq', label='CTLE fp (GHz)',        tooltip="CTLE peaking frequency (GHz)", ),
                        Item(name='peak_mag',  label='CTLE boost (dB)',      tooltip="CTLE peaking magnitude (dB)", ),
                        Item(name='ctle_offset', label='CTLE offset (dB)',   tooltip="CTLE d.c. offset (dB)", ),
                        Item(name='rx_bw',     label='Bandwidth (GHz)',      tooltip="unequalized signal path bandwidth (GHz).", ),
                    ),
                    VGroup(
                        Item(name='use_agc',   label='Use AGC',              tooltip="Include automatic gain control.", ),
                        Item(name='use_dfe',   label='Use DFE',              tooltip="Include DFE in simulation.", ),
                        Item(name='sum_ideal', label='Ideal DFE',            tooltip="Use ideal DFE. (performance boost)", ),
                    ),
                    label='Rx Equalization', show_border=True,
                ),
            ),
            HGroup(
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
            ),
            label = 'Config.', id = 'config',
        ),
        Group(
            Item('plots_dfe', editor=ComponentEditor(), show_label=False,),
            label = 'DFE', id = 'plots_dfe'
        ),
        VGroup(
            HGroup(
                VGroup(
                    HGroup(
                        Item(name='pretap_tune_enable',   label='Enable',                            tooltip="Enable this tap.", ),
                        Item(name='pretap_tune',          label='Pre-tap',     format_str='%7.4f',   tooltip="pre-cursor tap weight", ),
                    ),
                    HGroup(
                        Item(name='posttap_tune_enable',  label='Enable',                            tooltip="Enable this tap.", ),
                        Item(name='posttap_tune',         label='Post-tap',    format_str='%7.4f',   tooltip="post-cursor tap weight", ),
                    ),
                    HGroup(
                        Item(name='posttap2_tune_enable', label='Enable',                            tooltip="Enable this tap.", ),
                        Item(name='posttap2_tune',        label='Post-tap2',   format_str='%7.4f',   tooltip="2nd post-cursor tap weight", ),
                    ),
                    HGroup(
                        Item(name='posttap3_tune_enable', label='Enable',                            tooltip="Enable this tap.", ),
                        Item(name='posttap3_tune',        label='Post-tap3',   format_str='%7.4f',   tooltip="3rd post-cursor tap weight", ),
                    ),
                    label='Tx Equalization', show_border=True,
                ),
                VGroup(
                    Item(name='peak_freq_tune', label='CTLE fp (GHz)',                            tooltip="CTLE peaking frequency (GHz)", ),
                    Item(name='peak_mag_tune',  label='CTLE boost (dB)',      format_str='%7.4f', tooltip="CTLE peaking magnitude (dB)", ),
                    Item(name='rx_bw_tune',     label='Bandwidth (GHz)',                          tooltip="unequalized signal path bandwidth (GHz).", ),
                    Item(label="Note: Only peaking magnitude will be optimized;\nplease, set peak frequency and bandwidth appropriately."),
                    label = 'Rx Equalization', show_border = True, 
                ),
                VGroup(
                    Item(name='ideal_type',     label='Ideal Response Type',                      tooltip="Ideal impulse response type.",
                                                editor=CheckListEditor(values=[(0, 'Impulse'), (1, 'Sinc'), (2, 'Raised Cosine'),])),
                    Item(name='pulse_tune',     label='Use Pulse Response',                       tooltip="Use pulse response, as opposed to impulse response, to tune equalization.", ),
                    Item(name='max_iter',       label='Max. Iterations',                          tooltip="Maximum number of iterations to allow, during optimization.", ),
                    Item(name='rel_opt', label='Rel. Opt.', format_str='%7.4f',
                         tooltip="Relative optimization metric.", enabled_when='False'),
                    label = 'Tuning Options', show_border = True,
                ),
            ),
            Item('plot_h_tune', editor=ComponentEditor(), show_label=False,),
            HGroup(
                Item('btn_rst_eq',  show_label=False, tooltip="Reset all values to those on the 'Config.' tab.",),
                Item('btn_save_eq', show_label=False, tooltip="Store all values to 'Config.' tab.",),
                Item('btn_opt_tx',  show_label=False, tooltip="Run Tx tap weight optimization.",),
                Item('btn_opt_rx',  show_label=False, tooltip="Run Rx CTLE optimization.",),
                Item('btn_coopt',   show_label=False, tooltip="Run co-optimization.",),
            ),
            label = 'EQ Tune', id = 'eq_tune',
        ),
        Group(
            Item('plots_h', editor=ComponentEditor(), show_label=False,),
            label = 'Impulses', id = 'plots_h'
        ),
        Group(
            Item('plots_s', editor=ComponentEditor(), show_label=False,),
            label = 'Steps', id = 'plots_s'
        ),
        Group(
            Item('plots_p', editor=ComponentEditor(), show_label=False,),
            label = 'Pulses', id = 'plots_p'
        ),
        Group(
            Item('plots_H', editor=ComponentEditor(), show_label=False,),
            label = 'Freq. Resp.', id = 'plots_H'
        ),
        Group(
            Item('plots_out', editor=ComponentEditor(), show_label=False,),
            label = 'Outputs', id = 'plots_out'
        ),
        Group(
            Item('plots_eye', editor=ComponentEditor(), show_label=False,),
            label = 'Eyes', id = 'plots_eye'
        ),
        Group(
            Item('plots_jitter_dist', editor=ComponentEditor(), show_label=False,),
            label = 'Jitter Dist.', id = 'plots_jitter_dist'
        ),
        Group(
            Item('plots_jitter_spec', editor=ComponentEditor(), show_label=False,),
            label = 'Jitter Spec.', id = 'plots_jitter_spec'
        ),
        Group(
            Item('plots_bathtub', editor=ComponentEditor(), show_label=False,),
            label = 'Bathtubs', id = 'plots_bathtub'
        ),
        Group(
            Item('jitter_info', style='readonly', show_label=False),
            label = 'Jitter Info'
        ),
        Group(
            Item('sweep_info', style='readonly', show_label=False),
            label = 'Sweep Info'
        ),
        Group(
            Item('ident', style='readonly', show_label=False),
            Item('perf_info', style='readonly', show_label=False),
            label = 'About'
        ),
        Group(
            Item('instructions', style='readonly', show_label=False),
            label = 'Help'
        ),
        layout = 'tabbed', springy = True, id = 'tabs',
    ),
    resizable = True,
    handler = MyHandler(),
    buttons = [run_simulation, ],
    statusbar = "status_str",
    title='PyBERT',
    width=1200, height=800
)

