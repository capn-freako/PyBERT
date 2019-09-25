"""
Initialization file for Simbeor solver.

.. moduleauthor:: David Banas <capn.freako@gmail.com>

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   20 September 2019

Initialization of the Simbeor solver consists of:

* Finding the installation directory, using the `SIMBEOR_SDK` environment variable.

* Importing all the Python modules there.

Copyright (c) 2019 by David Banas; all rights reserved World wide.
"""
import os
import os.path       as osp
import pybert.solver as slvr

sdkdir   = os.environ.get('SIMBEOR_SDK')
__path__ = [osp.join(sdkdir, "python")]
__all__  = ["simbeor",]                     # Should contain the name of each submodule.
from . import *                             # Makes each submodule available as: pybert.solvers.simbeor.simbeor, etc.

class Solver(slvr.Solver):
    """Simbeor-specific override of `pybert.solver.Solver` class."""

    def __init__(self):
        super().__init__()
    
    def solve(self,
          ch_type    : slvr.ChType = "microstrip_se",  #: Channel cross-sectional configuration.
          diel_const : float  = 4.0,    #: Dielectric constant of substrate (rel.).
          thickness  : float  = 0.036,  #: Trace thickness (mm).
          width      : float  = 0.254,  #: Trace width (mm).
          height     : float  = 0.127,  #: Trace height above/below ground plane (mm).
          separation : float  = 0.508,  #: Trace separation (mm).
          roughness  : float  = 0.004,  #: Trace surface roughness (mm-rms).
          lic_path   : str    = "C:\\Users\\dbanas\\Downloads\\simbeor_DavidBanas_09152019.lic",
          lic_name   : str    = "simbeor_complete",
          prj_name   : str    = "SimbeorPyBERT"
         ):
        """Use the simbeor.pyd Python library to solve the channel."""
        
        # Make sure the `simbeor.pyd` library doesn't attempt to open a console, and initialize.
        cfg = simbeor.GetSimbeorOptions()
        cfg['KeepConsoleWindow'] = False
        simbeor.SetSimbeorOptions(cfg)
        simbeor.Initialize()

        # Make sure we've got a license.
        if(not simbeor.SetLicense(lic_path, lic_name)):
            raise RuntimeError("Couldn't get Simbeor license!")

        # Setup Simbeor project.
        prj = simbeor.ProjectCreate(prj_name, True, True, 20.0) # Project with materials and stackup sections.
        if prj == 0:
            return simbeor.CheckResult("Create Project")
        # - Setup materials.
        #Wideband Debye aka Djordjevic-Sarkar, best for PCBs and packages
        fr4 = simbeor.MaterialAddDielectric(osp.join(prj_name, "FR4"), 4.3, 0.02, 1.0e9, {}, 1e5, 1e12)
        if fr4 == 0:
            return simbeor.CheckResult("Add material")
        air = simbeor.MaterialAddDielectric(osp.join(prj_name, "Air"), 1.0, 0.0, 1e9, {}, 1e5, 1e12)
        if air == 0:
            return simbeor.CheckResult("Add material")
        rough = simbeor.InitRoughness()
        rough['RoughnessModel'] = "HurayBracken"
        rough['SR'] = 0.0
        rough['RF'] = 8.0
        cop = simbeor.MaterialAddConductor(osp.join(prj_name, "Copper"), 1.0, rough, 0.004)
        if cop == 0:
            return simbeor.CheckResult("Add material")
        # - Setup stack-up.
        sig1 = simbeor.LayerAddSignal(osp.join(prj_name, "TOP"), "Copper", "Air", thickness * 1.0e-3)
        if sig1 == 0:
            return simbeor.CheckResult("Add layer")

        med1 = simbeor.LayerAddMedium(osp.join(prj_name, "core"), "FR4", height * 1.0e-3)
        if med1 == 0:
            return simbeor.CheckResult("Add layer")

        pln1 = simbeor.LayerAddPlane(osp.join(prj_name, "Plane1"), "Copper", "FR4", thickness * 1.0e-3)
        if pln1 == 0:
            return simbeor.CheckResult("Add layer")

        # Access to default signal configuration.
        scfg = simbeor.InitSignalConfigurator()  # easy way to configure signals and sweeps
        simbeor.ConfigureDefaultSignals(scfg)    # Define bit rate and rise time here.

        # Setup transmission line properties, as per the parameters we've been called with.
        # (Note that Simbeor library expects units of meters.)
        tline = simbeor.InitSingleTLine()
        tline['Width']      = width * 1.0e-3
        tline['StripShape'] = "Trapezoidal"
        tline['EtchFactor'] = 0.0
        tline['Clearance']  = 0.0                   # Non-zero for coplanar only.
        tline['LayerName']  = "TOP"                 # Surface layer.

        # Calculate Z0.
        zresult, result = simbeor.CalcSingleTLine_Z(prj_name, tline)
        simbeor.CheckResult(result, "calculating Zo")
        print(tline, zresult, "Forward")

        # Calculate frequency-dependent loss.
        # - Examples from Simbeor `test_zcalc.py` file.
        # def TestModelSingle(ProjectName, ModelName, LayerName, CoreBelow, Coplanar, SolutionDirectory):
        # TestModelSingle(Project1M, "Project(1M)\\SingleMSL", "TOP", True, False, '' ) #model for single microstrip        
        # - Optionally, define frequency sweep. Otherwise, sweep is defined by the signal configurator
        frqSweep = simbeor.GetDefaultFrequencySweep() #access to default sweep
        opt = simbeor.GetDefault_SFS_Options()
        # - Build model and simulate
        ModelName = osp.join(prj_name, "SingleMSL")
        result = simbeor.ModelSingleTLine_SFS(ModelName, tline, frqSweep, opt) #frqSweep is not needed if common default sweep is used, opt are not needed most of the time
        simbeor.CheckResult(result, "single t-line model building with SFS")
        
        if result == 0:
            #TEST ACCESS TO GAMMA, Z0
            frqCount = simbeor.GetFrequencyPointsCount(ModelName) #get number of computed frequency points
            pfrqs = simbeor.GetFrequencyPoints(ModelName) #get all frequency points
            if len( pfrqs ) == frqCount:
                pAtt = simbeor.GetPropagationConstants(ModelName, 'DBAttenuation', 1) #get attenuation in dB/m into pAtt array
                if len( pAtt ) != frqCount:
                    simbeor.CheckResult("propagation constant")
                pZo = simbeor.GetCharacteristicImpedances(ModelName, 'Magnitude', 1) #get characteristic impedance in Ohm into pAtt array
                if len( pZo ) != frqCount:
                    simbeor.CheckResult("characteristic impedance")

                #ACCESS TO PER UNIT LENGTH RLGC PARAMETERS
                for i in range(frqCount):
                    R, L, G, C, result = simbeor.GetRLGC(ModelName, pfrqs[i], 1)
                    if result != 0:
                        simbeor.CheckResult(result, "RLGC of 1-conductor t-line")
                    #fill arrays of R,L,G,C if necessary
                
                # if SolutionDirectory != '':
                #     #EXAMPLE FO SAVING W-ELEMENT MODEL INTO FILE
                #     fileName = SolutionDirectory + '\\' + ModelName
                #     fileName += ".sp"
                #     result = simbeor.SaveTLineModelToFile(ModelName, fileName, True)
                #     simbeor.CheckResult(result, "Save W-element model of 1-conductor t-line")
            else:
                simbeor.checkResult('SimulationFailed', "Get frequency points")

        simbeor.Cleanup()
        simbeor.Uninitialize()
        return (pfrqs, pAtt)
        