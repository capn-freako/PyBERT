"""
Default view definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from `pybert.py', as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

import pickle

from threading import Thread

from pyface.api import FileDialog, OK
from traits.api import Instance
from traitsui.api import View, Item, Group, VGroup, HGroup, Action, Handler, \
                         DefaultOverride, CheckListEditor, StatusItem, \
                         TextEditor, TableEditor, ObjectColumn, spring, TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from enable.component_editor import ComponentEditor

from chaco.api import ArrayPlotData

from pybert_cntrl import my_run_sweeps
from pybert_data  import PyBertData

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

    def do_save_data(self, info):
        the_pybert = info.object
        dlg = FileDialog(action='save as', wildcard='*.pybert_data', default_path=the_pybert.data_file)
        if dlg.open() == OK:
            try:
                plotdata = PyBertData(the_pybert)
                with open(dlg.path, 'wt') as the_file:
                    pickle.dump(plotdata, the_file)
                the_pybert.data_file = dlg.path
            except Exception as err:
                err.message = "The following error occured:\n\t{}\nThe waveform data was NOT saved.".format(err.message)
                the_pybert.handle_error(err)

    def do_load_data(self, info):
        the_pybert = info.object
        dlg = FileDialog(action='open', wildcard='*.pybert_data', default_path=the_pybert.data_file)
        if dlg.open() == OK:
            try:
                with open(dlg.path, 'rt') as the_file:
                    the_plotdata = pickle.load(the_file)
                if(type(the_plotdata) is not PyBertData):
                    raise Exception("The data structure read in is NOT of type: ArrayPlotData!")
                for prop, value in the_plotdata.the_data.arrays.iteritems():
                    the_pybert.plotdata.set_data(prop + '_ref', value)
                the_pybert.data_file = dlg.path

                # Add reference plots, if necessary.
                # - time domain
                for (container, suffix, has_both) in [
                        (the_pybert.plots_h.component_grid.flat, 'h', False),
                        (the_pybert.plots_s.component_grid.flat, 's', True),
                        (the_pybert.plots_p.component_grid.flat, 'p', False),
                        ]:
                    if(not 'Reference' in container[0].plots):
                        (ix, prefix) = (0, 'chnl')
                        item_name = prefix + "_" + suffix + "_ref"
                        container[ix].plot(("t_ns_chnl", item_name),
                                type="line", color="darkcyan",  name="Inc_ref")
                        for (ix, prefix) in [
                                (1, 'tx'),
                                (2, 'ctle'),
                                (3, 'dfe'), ]:
                            item_name = prefix + "_out_" + suffix + "_ref"
                            container[ix].plot(("t_ns_chnl", item_name),
                                    type="line", color="darkmagenta",  name="Cum_ref")
                        if(has_both):
                            for (ix, prefix) in [
                                    (1, 'tx'),
                                    (2, 'ctle'),
                                    (3, 'dfe'), ]:
                                item_name = prefix + "_" + suffix + "_ref"
                                container[ix].plot(("t_ns_chnl", item_name),
                                        type="line", color="darkcyan",  name="Inc_ref")

                # - frequency domain
                for (container, suffix, has_both) in [
                        (the_pybert.plots_H.component_grid.flat, 'H', True)
                        ]:
                    if(not 'Reference' in container[0].plots):
                        (ix, prefix) = (0, 'chnl')
                        item_name = prefix + "_" + suffix + "_ref"
                        container[ix].plot(("f_GHz", item_name),
                                type="line", color="darkcyan",  name="Inc_ref", index_scale='log')
                        for (ix, prefix) in [
                                (1, 'tx'),
                                (2, 'ctle'),
                                (3, 'dfe'), ]:
                            item_name = prefix + "_out_" + suffix + "_ref"
                            container[ix].plot(("f_GHz", item_name),
                                    type="line", color="darkmagenta",  name="Cum_ref", index_scale='log')
                        if(has_both):
                            for (ix, prefix) in [
                                    (1, 'tx'),
                                    (2, 'ctle'),
                                    (3, 'dfe'), ]:
                                item_name = prefix + "_" + suffix + "_ref"
                                container[ix].plot(("f_GHz", item_name),
                                        type="line", color="darkcyan",  name="Inc_ref", index_scale='log')

            except Exception as err:
                print item_name
                err.message = "The following error occured:\n\t{}\nThe waveform data was NOT loaded.".format(err.message)
                the_pybert.handle_error(err)

run_simulation = Action(name="Run",     action="do_run_simulation")
save_data = Action(name="Save Results", action="do_save_data")
load_data = Action(name="Load Results", action="do_load_data")
    
# Main window layout definition.
traits_view = View(
    Group(
        VGroup(
            HGroup(
                HGroup(
                    VGroup(
                        Item(name='bit_rate',    label='Bit Rate (Gbps)',  tooltip="bit rate", show_label=True, enabled_when='True',
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)
                        ),
                        Item(name='nbits',       label='Nbits',    tooltip="# of bits to run",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=int)
                        ),
                        Item(name='nspb',        label='Nspb',     tooltip="# of samples per bit",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=int)
                        ),
                        Item(name='mod_type',    label='Modulation', tooltip="line signalling/modulation scheme",
                            editor=CheckListEditor(values=[(0, 'NRZ'), (1, 'Duo-binary'), (2, 'PAM-4'),])
                        ),
                    ),
                    VGroup(
                        Item(name='do_sweep',    label='Do Sweep',    tooltip="Run parameter sweeps.", ),
                        Item(name='sweep_aves',  label='SweepAves',   tooltip="# of trials, per sweep, for averaging.", enabled_when='do_sweep == True'),
                        Item(name='pattern_len', label='PatLen',   tooltip="length of random pattern to use to construct bit stream",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=int)
                        ),
                        Item(name='eye_bits',    label='EyeBits',  tooltip="# of bits to use to form eye diagrams",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=int)
                        ),
                    ),
                    VGroup(
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
                        Item(name='impulse_length', label='Impl. Len. (ns)', tooltip="Manual impulse response length override", ),
                        Item(name='cac',     label='Rx_Cac (uF)',    enabled_when='use_ch_file == False', tooltip="Rx a.c. coupling capacitance (each pin)", ),
                    ),
                    VGroup(
                        Item(name='Theta0',  label='Loss Tan.',   enabled_when='use_ch_file == False', tooltip="dielectric loss tangent", ),
                        Item(name='Z0',      label='Z0 (Ohms)',   enabled_when='use_ch_file == False', tooltip="characteristic differential impedance", ),
                        Item(name='v0',      label='v_rel (c)',   enabled_when='use_ch_file == False', tooltip="normalized propagation velocity", ),
                        Item(name='l_ch',    label='Length (m)',  enabled_when='use_ch_file == False', tooltip="interconnect length", ),
                    ),
                    VGroup(
                        Item(name='rs',      label='Tx_Rs (Ohms)',   enabled_when='use_ch_file == False', tooltip="Tx differential source impedance", ),
                        Item(name='cout',    label='Tx_Cout (pF)',   enabled_when='use_ch_file == False', tooltip="Tx parasitic output capacitance (each pin)", ),
                        Item(name='rin',     label='Rx_Rin (Ohms)',  enabled_when='use_ch_file == False', tooltip="Rx differential input impedance", ),
                        Item(name='cin',     label='Rx_Cin (pF)',    enabled_when='use_ch_file == False', tooltip="Rx parasitic input capacitance (each pin)", ),
                    ),
                    label='Channel Parameters', show_border=True,
                ),
            ),
            HGroup(
                VGroup(
                    VGroup(
                        HGroup(
                            VGroup(
                                HGroup(
                                    Item(name='tx_ami_valid', show_label=False, style='simple', enabled_when='False'),
                                    Item(name='tx_ami_file', label='AMI File:',    tooltip="Choose AMI file."),
                                ),
                                HGroup(
                                    Item(name='tx_dll_valid', show_label=False, style='simple', enabled_when='False'),
                                    Item(name='tx_dll_file', label='DLL File:',    tooltip="Choose DLL file."),
                                ),
                            ),
                            VGroup(
                                Item(name='tx_use_ami',      label='Use AMI',      tooltip="You must select both files, first.",
                                    enabled_when='tx_ami_valid == True and tx_dll_valid == True'),
                                Item(name='tx_use_getwave',  label='Use GetWave',  tooltip="Use the model's GetWave() function.",
                                    enabled_when='tx_use_ami and tx_has_getwave'),
                                Item('btn_cfg_tx',  show_label=False, tooltip="Configure Tx AMI parameters.",
                                    enabled_when='tx_ami_valid == True'),
                            ),
                        ),
                        label='IBIS-AMI', show_border=True,
                    ),
                    VGroup(
                        Item(   name='tx_taps',
                                editor=TableEditor(columns=[ObjectColumn(name='name', editable=False),
                                                            ObjectColumn(name='enabled', style='simple'),
                                                            ObjectColumn(name='min_val', horizontal_alignment='center'),
                                                            ObjectColumn(name='max_val', horizontal_alignment='center'),
                                                            ObjectColumn(name='value', format='%+05.3f', horizontal_alignment='center'),
                                                            ObjectColumn(name='steps', horizontal_alignment='center'),
                                                           ],
                                                    configurable=False,
                                                    reorderable=False,
                                                    sortable=False,
                                                    selection_mode='cell',
                                                    auto_size=True,
                                                    rows=4,
                                                    # v_size_policy='ignored',
                                                    # h_size_policy='minimum',
                                                    # orientation='vertical',
                                                    # is_grid_cell=True,
                                                    # show_toolbar=False,
                                                   ),
                                show_label=False,
                        ),
                        label='Native', show_border=True,
                        enabled_when='tx_use_ami == False'
                    ),
                    label='Tx Equalization', show_border=True,
                ),
                VGroup(
                    VGroup(
                        HGroup(
                            VGroup(
                                HGroup(
                                    Item(name='rx_ami_valid', show_label=False, style='simple', enabled_when='False'),
                                    Item(name='rx_ami_file', label='AMI File:',    tooltip="Choose AMI file."),
                                ),
                                HGroup(
                                    Item(name='rx_dll_valid', show_label=False, style='simple', enabled_when='False'),
                                    Item(name='rx_dll_file', label='DLL File:',    tooltip="Choose DLL file."),
                                ),
                            ),
                            VGroup(
                                Item(name='rx_use_ami',      label='Use AMI',      tooltip="You must select both files, first.",
                                    enabled_when='rx_ami_valid == True and rx_dll_valid == True'),
                                Item(name='rx_use_getwave',  label='Use GetWave',  tooltip="Use the model's GetWave() function.",
                                    enabled_when='rx_use_ami and rx_has_getwave'),
                                Item('btn_cfg_rx',  show_label=False, tooltip="Configure Rx AMI parameters.",
                                    enabled_when='rx_ami_valid == True'),
                            ),
                        ),
                        label='IBIS-AMI', show_border=True,
                    ),
                    HGroup(
                        VGroup(
                            HGroup(
                                Item(name='use_ctle_file', label='fromFile', tooltip='Select CTLE impulse/step response from file.', ),
                                Item(name='ctle_file', label='Filename',    enabled_when='use_ctle_file == True'),
                            ),
                            HGroup(
                                Item(name='peak_freq', label='CTLE fp (GHz)',   tooltip="CTLE peaking frequency (GHz)",
                                        enabled_when='use_ctle_file == False' ),
                                Item(name='rx_bw',     label='Bandwidth (GHz)', tooltip="unequalized signal path bandwidth (GHz).",
                                        enabled_when='use_ctle_file == False' ),
                            ),
                            HGroup(
                                Item(name='peak_mag',  label='CTLE boost (dB)', tooltip="CTLE peaking magnitude (dB)",
                                    format_str='%4.1f', enabled_when='use_ctle_file == False' ),
                                Item(name='ctle_mode', label='CTLE mode', tooltip="CTLE Operating Mode", enabled_when='use_ctle_file == False' ),
                                Item(name='ctle_offset', tooltip="CTLE d.c. offset (dB)",
                                        show_label=False, enabled_when='ctle_mode == "Manual"'),
                            ),
                        ),
                        label='Native', show_border=True,
                        enabled_when='rx_use_ami == False'
                    ),
                    label='Rx Equalization', show_border=True,
                ),
            ),
            HGroup(
                HGroup(
                    VGroup(
                        Item(name='delta_t',      label='Delta-t (ps)',  tooltip="magnitude of CDR proportional branch", ),
                        Item(name='alpha',        label='Alpha',         tooltip="relative magnitude of CDR integral branch", ),
                        Item(name='n_lock_ave',   label='Lock Nave.',    tooltip="# of UI estimates to average, when determining lock", ),
                    ),
                    VGroup(
                        Item(name='rel_lock_tol', label='Lock Tol.',     tooltip="relative tolerance for determining lock", ),
                        Item(name='lock_sustain', label='Lock Sus.',     tooltip="length of lock determining hysteresis vector", ),
                    ),
                    label='CDR Parameters', show_border=True,
                    # enabled_when='rx_use_ami == False  or  rx_use_ami == True and rx_use_getwave == False',
                ),
                HGroup(
                    VGroup(
                        Item(name='gain',            label='Gain',   tooltip="error feedback gain", ),
                        Item(name='n_taps',          label='Taps',   tooltip="# of taps", ),
                        Item(name='decision_scaler', label='Level',  tooltip="target output magnitude", ),
                        enabled_when='use_dfe == True',
                    ),
                    VGroup(
                        VGroup(
                            Item(name='n_ave',           label='Nave.',    tooltip="# of CDR adaptations per DFE adaptation", ),
                            Item(name='sum_bw',          label='BW (GHz)', tooltip="summing node bandwidth", enabled_when='sum_ideal == False'),
                            enabled_when='use_dfe == True',
                        ),
                        HGroup(
                            Item(name='use_dfe',   label='Use DFE',   tooltip="Include DFE in simulation.", ),
                            Item(name='sum_ideal', label='Ideal DFE', tooltip="Use ideal DFE. (performance boost)", enabled_when='use_dfe == True', ),
                        ),
                    ),
                    label='DFE Parameters', show_border=True,
                    # enabled_when='rx_use_ami == False  or  rx_use_ami == True and rx_use_getwave == False',
                ),
                VGroup(
                    Item(name='thresh',          label='Pj Thresh.',   tooltip="Threshold for identifying periodic jitter spectral elements. (sigma)", ),
                    label='Analysis Parameters', show_border=True,
                ),
            ),
            HGroup(
                Item('btn_save_cfg',  show_label=False, tooltip="Store the current simulation configuration data to a file.",),
                Item('btn_load_cfg',  show_label=False, tooltip="Load the simulation configuration data from a file.",),
                Item('cfg_file',      show_label=False, style='readonly'),
            ),
            spring,
            label = 'Config.', id = 'config',
        ),
        Group(
            Item('console_log', show_label=False, style='custom'),
            label = 'Console', id = 'console'
        ),
        Group(
            Item('plots_dfe', editor=ComponentEditor(), show_label=False,),
            label = 'DFE', id = 'plots_dfe'
        ),
        # "EQ Tune" tab.
        VGroup(
            HGroup(
                Group(
                    Item(   name='tx_tap_tuners',
                            editor=TableEditor(columns=[ObjectColumn(name='name', editable=False),
                                                        ObjectColumn(name='enabled'),
                                                        ObjectColumn(name='min_val'),
                                                        ObjectColumn(name='max_val'),
                                                        ObjectColumn(name='value', format='%+05.3f'),
                                                       ],
                                                configurable=False,
                                                reorderable=False,
                                                sortable=False,
                                                selection_mode='cell',
                                                auto_size=True,
                                                rows=4,
                                                orientation='horizontal',
                                                is_grid_cell=True,
                                               ),
                            show_label=False,
                        ),
                    label='Tx Equalization', show_border=True,
                ),
                HGroup(
                    VGroup(
                        Item(name='peak_freq_tune', label='CTLE fp (GHz)',   tooltip="CTLE peaking frequency (GHz)", ),
                        Item(name='rx_bw_tune', label='Bandwidth (GHz)', tooltip="unequalized signal path bandwidth (GHz).", ),
                        Item(name='peak_mag_tune', label='CTLE boost (dB)', tooltip="CTLE peaking magnitude (dB)",
                             format_str='%4.1f'),
                        HGroup(
                            Item(name='ctle_mode_tune', label='CTLE mode', tooltip="CTLE Operating Mode"),
                            Item(name='ctle_offset_tune', tooltip="CTLE d.c. offset (dB)",
                                    show_label=False, enabled_when='ctle_mode_tune == "Manual"'),
                        ),
                        HGroup(
                            Item(name='use_dfe_tune', label='Use DFE.', tooltip="Include ideal DFE in optimization."),
                            Item(name='n_taps_tune',  label='Taps',     tooltip="Number of DFE taps."),
                        ),
                    ),
                    Item(label="Note: Only peaking magnitude\nwill be optimized; please, set\npeak frequency, bandwidth, and\nmode appropriately."),
                    label='Rx Equalization', show_border=True,
                ),
                VGroup(
                    Item(   name='max_iter', label='Max. Iterations',
                            tooltip="Maximum number of iterations to allow, during optimization.", ),
                    Item(   name='rel_opt', label='Rel. Opt.:', format_str='%7.4f',
                            tooltip="Relative optimization metric.", style='readonly'),
                    Item(   name='przf_err', label='PRZF Err.:', format_str='%5.3f',
                            tooltip="Pulse Response Zero Forcing approximation error.", style='readonly'),
                    label = 'Tuning Options', show_border = True,
                ),
                springy=False,
            ),
            Item('plot_h_tune', editor=ComponentEditor(), show_label=False,
                                springy=True),
            HGroup(
                Item('btn_rst_eq',  show_label=False, tooltip="Reset all values to those on the 'Config.' tab.",),
                Item('btn_save_eq', show_label=False, tooltip="Store all values to 'Config.' tab.",),
                Item('btn_opt_tx',  show_label=False, tooltip="Run Tx tap weight optimization.",),
                Item('btn_opt_rx',  show_label=False, tooltip="Run Rx CTLE optimization.",),
                Item('btn_coopt',   show_label=False, tooltip="Run co-optimization.",),
                Item('btn_abort',   show_label=False, tooltip="Abort all optimizations.",),
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
    resizable = False,
    handler = MyHandler(),
    buttons = [run_simulation, save_data, load_data],
    statusbar = "status_str",
    title='PyBERT',
    width=0.95, height=0.95
)

