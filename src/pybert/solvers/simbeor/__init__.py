# pylint: disable=undefined-variable
# type: ignore
# flake8: noqa

"""Initialization file for Simbeor solver.

.. moduleauthor:: David Banas <capn.freako@gmail.com>

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   20 September 2019

Initialization of the Simbeor solver consists of:

* Finding the installation directory, using the `SIMBEOR_SDK` environment variable.

* Importing all the Python modules there.

Copyright (c) 2019 by David Banas; all rights reserved World wide.
"""
import os
import os.path as osp
from typing import List

import numpy as np

from pybert.solvers import solver as slvr

sdkdir = os.environ.get("SIMBEOR_SDK")
__path__ = [osp.join(sdkdir, "python")]
# __all__ = [
#     "simbeor",
# ]  # Should contain the name of each submodule.
# from . import *  # Makes each submodule available as: pybert.solvers.simbeor.simbeor, etc.


class Solver(slvr.Solver):  # pylint: disable=too-few-public-methods
    """Simbeor-specific override of `pybert.solver.Solver` class."""

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements,unused-argument,too-many-positional-arguments
    def solve(
        self,
        ch_type: slvr.ChType = "microstrip_se",  #: Channel cross-sectional configuration.
        diel_const: float = 4.0,  #: Dielectric constant of substrate (rel.).
        loss_tan: float = 0.02,  #: Loss tangent at ``des_freq``.
        des_freq: float = 1.0e9,  #: Frequency at which ``diel_const`` and ``loss_tan`` are quoted (Hz).
        thickness: float = 0.036,  #: Trace thickness (mm).
        width: float = 0.254,  #: Trace width (mm).
        height: float = 0.127,  #: Trace height above/below ground plane (mm).
        separation: float = 0.508,  #: Trace separation (mm).
        roughness: float = 0.004,  #: Trace surface roughness (mm-rms).
        fs: List[float] = None,  #: Angular frequency sample points (Hz).
        lic_path: str = "",
        lic_name: str = "simbeor_complete",
        prj_name: str = "SimbeorPyBERT",
    ):
        """Use the simbeor.pyd Python library to solve the channel."""

        # Make sure the `simbeor.pyd` library doesn't attempt to open a console, and initialize.
        cfg = simbeor.GetSimbeorOptions()
        cfg["KeepConsoleWindow"] = False
        simbeor.SetSimbeorOptions(cfg)
        simbeor.Initialize()

        # Make sure we've got a license.
        if not simbeor.SetLicense(lic_path, lic_name):
            raise RuntimeError("Couldn't get Simbeor license!")

        # Setup Simbeor project.
        prj = simbeor.ProjectCreate(prj_name, True, True, 20.0)  # Project with materials and stackup sections.
        if prj == 0:
            raise RuntimeError("Couldn't create a Simbeor project: " + simbeor.GetErrorMessage())

        # - Setup materials.
        # Wideband Debye aka Djordjevic-Sarkar, best for PCBs and packages
        fr4 = simbeor.MaterialAddDielectric(osp.join(prj_name, "FR4"), diel_const, loss_tan, des_freq, {}, 1e5, 1e12)
        if fr4 == 0:
            raise RuntimeError("Couldn't create a dielectric material: " + simbeor.GetErrorMessage())
        air = simbeor.MaterialAddDielectric(osp.join(prj_name, "Air"), 1.0, 0.0, 1e9, {}, 1e5, 1e12)
        if air == 0:
            raise RuntimeError("Couldn't create air material: " + simbeor.GetErrorMessage())
        rough = simbeor.InitRoughness()
        rough["RoughnessModel"] = "HurayBracken"
        rough["SR"] = 0.0
        rough["RF"] = 8.0
        cop = simbeor.MaterialAddConductor(osp.join(prj_name, "Copper"), 1.0, rough, 0.004)
        if cop == 0:
            raise RuntimeError("Couldn't create a conductor material: " + simbeor.GetErrorMessage())

        # - Setup stack-up.
        if ch_type in ("stripline_se", "stripline_diff"):  # stripline
            pln0 = simbeor.LayerAddPlane(osp.join(prj_name, "Plane0"), "Copper", "FR4", thickness * 1.0e-3)
            if pln0 == 0:
                raise RuntimeError("Couldn't create a plane layer: " + simbeor.GetErrorMessage())
            med0 = simbeor.LayerAddMedium(osp.join(prj_name, "prepreg"), "FR4", height * 1.0e-3)
            if med0 == 0:
                raise RuntimeError("Couldn't create a dielectric layer: " + simbeor.GetErrorMessage())
            sig1 = simbeor.LayerAddSignal(osp.join(prj_name, "TOP"), "Copper", "FR4", thickness * 1.0e-3)
            if sig1 == 0:
                raise RuntimeError("Couldn't create a signal layer: " + simbeor.GetErrorMessage())
        else:  # microstrip
            sig1 = simbeor.LayerAddSignal(osp.join(prj_name, "TOP"), "Copper", "Air", thickness * 1.0e-3)
            if sig1 == 0:
                raise RuntimeError("Couldn't create a signal layer: " + simbeor.GetErrorMessage())
        med1 = simbeor.LayerAddMedium(osp.join(prj_name, "core"), "FR4", height * 1.0e-3)
        if med1 == 0:
            raise RuntimeError("Couldn't create a dielectric layer: " + simbeor.GetErrorMessage())
        pln1 = simbeor.LayerAddPlane(osp.join(prj_name, "Plane1"), "Copper", "FR4", thickness * 1.0e-3)
        if pln1 == 0:
            raise RuntimeError("Couldn't create a plane layer: " + simbeor.GetErrorMessage())

        # Access to default signal configuration.
        scfg = simbeor.InitSignalConfigurator()  # easy way to configure signals and sweeps
        simbeor.ConfigureDefaultSignals(scfg)  # Define bit rate and rise time here.

        # Setup transmission line properties, as per the parameters we've been called with.
        # (Note that Simbeor library expects units of meters.)
        if ch_type in ("stripline_se", "microstrip_se"):  # single-ended
            tline = simbeor.InitSingleTLine()
        else:  # differential
            tline = simbeor.InitDiffTLine()
            tline["Spacing"] = separation * 1.0e-3
        tline["Width"] = width * 1.0e-3
        tline["StripShape"] = "Trapezoidal"
        tline["EtchFactor"] = 0.0
        tline["Clearance"] = 0.0  # Non-zero for coplanar only.
        tline["LayerName"] = "TOP"  # Surface layer.

        # Calculate Z0.
        if ch_type in ("stripline_se", "microstrip_se"):  # single-ended
            zresult, result = simbeor.CalcSingleTLine_Z(prj_name, tline)
        else:
            zresult, result = simbeor.CalcDiffTLine_Z(prj_name, tline)
        # simbeor.CheckResult(result, "calculating Zo")
        print(tline, zresult, "Forward")

        # Calculate frequency-dependent loss.
        # - Examples from Simbeor `test_zcalc.py` file.
        # def TestModelSingle(ProjectName, ModelName, LayerName, CoreBelow, Coplanar, SolutionDirectory):
        # TestModelSingle(Project1M, "Project(1M)\\SingleMSL", "TOP", True, False, '' ) #model for single microstrip
        # - Optionally, define frequency sweep. Otherwise, sweep is defined by the signal configurator
        frqSweep = simbeor.GetDefaultFrequencySweep()  # access to default sweep
        frqSweep["Start"] = fs[0]
        frqSweep["Stop"] = fs[-1]
        frqSweep["Count"] = len(fs)
        frqSweep["SweepType"] = "Equidistant"
        simbeor.SetDefaultFrequencySweep(frqSweep)  # access to default sweep
        opt = simbeor.GetDefault_SFS_Options()

        # - Build model and simulate
        ModelName = osp.join(prj_name, "SingleMSL")
        if ch_type in ("stripline_se", "microstrip_se"):  # single-ended
            result = simbeor.ModelSingleTLine_SFS(
                ModelName, tline, frqSweep, opt
            )  # frqSweep is not needed if common default sweep is used, opt are not needed most of the time
        else:
            result = simbeor.ModelDiffTLine_SFS(ModelName, tline, frqSweep, opt)
        # simbeor.CheckResult(result, "single t-line model building with SFS")
        if result != 0:
            raise RuntimeError("Simbeor channel simulation failed: " + simbeor.GetErrorMessage())

        # - Sanity check the results.
        frqCount = simbeor.GetFrequencyPointsCount(ModelName)  # get number of computed frequency points
        if frqCount != len(fs):
            raise RuntimeError(f"Simbeor channel simulation returned wrong number of frequency points: {frqCount}")
        pfrqs = simbeor.GetFrequencyPoints(ModelName)  # get all frequency points
        if (pfrqs != fs).any():
            raise RuntimeError("Simbeor channel simulation returned different set of frequency points!")

        # - Assemble return vectors.
        alpha = np.array(simbeor.GetPropagationConstants(ModelName, "Attenuation", 1))
        beta = np.array(simbeor.GetPropagationConstants(ModelName, "PhaseConstant", 1))
        ZcR = np.array(simbeor.GetCharacteristicImpedances(ModelName, "Real", 1))
        ZcI = np.array(simbeor.GetCharacteristicImpedances(ModelName, "Imaginary", 1))

        # - Clean up and return.
        simbeor.Cleanup()
        simbeor.Uninitialize()
        return ((alpha + 1j * beta), (ZcR + 1j * ZcI), pfrqs)


solver = Solver()
