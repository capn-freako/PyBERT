"""
General IBIS-AMI utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

from numpy import append, array, convolve  # type: ignore

from pyibisami.ami.model import AMIModel, AMIModelInitializer
from pyibisami.ami.parser import AMIParamConfigurator

from pybert.common import *  # pylint: disable=wildcard-import,unused-wildcard-import  # noqa: F403


def getwave_step_resp(ami_model):
    """Use a model's AMI_GetWave() function to extract its step response.

    Args:
        ami_model (): The AMI model to use.

    Returns:
        NumPy 1-D array: The model's step response.

    Raises:
        RuntimeError: When no step rise is detected.
    """
    # Delay the input edge slightly, in order to minimize high
    # frequency artifactual energy sometimes introduced near
    # the signal edges by frequency domain processing in some models.
    tmp = array([-0.5] * 128 + [0.5] * 896)  # Stick w/ 2^n, for freq. domain models' sake.
    s, _ = ami_model.getWave(tmp)
    # Some models delay signal flow through GetWave() arbitrarily.
    tmp = array([0.5] * 1024)
    max_tries = 10
    n_tries = 0
    while max(s) < 0 and n_tries < max_tries:  # Wait for step to rise, but not indefinitely.
        s, _ = ami_model.getWave(tmp)
        n_tries += 1
    if n_tries == max_tries:
        raise RuntimeError("No step rise detected!")
    # Make one more call, just to ensure a sufficient "tail".
    tmp, _ = ami_model.getWave(tmp)
    s = append(s, tmp)
    return s - s[0]


def init_imp_resp(ami_model):
    """Use a model's AMI_Init() function to extract its impulse response.

    Args:
        ami_model (): The AMI model to use.

    Returns:
        NumPy 1-D array: The model's impulse response.
    """

    # Delay the input edge slightly, in order to minimize high
    # frequency artifactual energy sometimes introduced near
    # the signal edges by frequency domain processing in some models.
    tmp = array([-0.5] * 128 + [0.5] * 896)  # Stick w/ 2^n, for freq. domain models' sake.
    s, _ = ami_model.getWave(tmp)
    # Some models delay signal flow through GetWave() arbitrarily.
    tmp = array([0.5] * 1024)
    max_tries = 10
    n_tries = 0
    while max(s) < 0 and n_tries < max_tries:  # Wait for step to rise, but not indefinitely.
        s, _ = ami_model.getWave(tmp)
        n_tries += 1
    if n_tries == max_tries:
        raise RuntimeError("No step rise detected!")
    # Make one more call, just to ensure a sufficient "tail".
    tmp, _ = ami_model.getWave(tmp)
    s = append(s, tmp)
    return s - s[0]


# pylint: disable=too-many-arguments,too-many-locals
def run_ami_model(dll_fname: str, param_cfg: AMIParamConfigurator, use_getwave: bool,
                  ui: float, ts: float, chnl_h: Rvec, x: Rvec, bits_per_call: int = 0  # noqa: F405
                  ) -> tuple[Rvec, Rvec, Rvec, Rvec, str]:  # noqa: F405
    """
    Run a simulation of an IBIS-AMI model.

    Args:
        dll_fname: Filename of DLL/SO.
        param_cfg: A pre-configured ``AMIParamConfigurator`` instance.
        use_getwave: Use ``AMI_GetWave()`` when True, ``AMI_Init()`` when False.
        ui: Unit interval (s).
        ts: Sample interval (s).
        chnl_h: Impulse response input to model (V/sample).
        x: Input waveform.

    Keyword Args:
        bits_per_call: Number of bits per call of `GetWave()`.
            Default: 0 (Means "Use existing value.")

    Returns:
        y, clks, h, out_h, params_out: A tuple consisting of:
            - the model output convolved w/ any channel impulse response given in `chnl_h`,
            - the model determined sampling instants (a.k.a. - "clock times"), if appropriate,
            - the model's impulse response (V/sample),
            - the impulse response of the model concatenated w/ the given channel (V/sample), and
            - input parameters, and any output parameters and/or message returned by the model.

    Raises:
        IOError: if the given file name cannot be found/opened.
        RuntimeError: if the given model doesn't support the requested mode.
    """

    # Validate the model against the requested use mode.
    if use_getwave:
        assert param_cfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]), RuntimeError(
            "You've requested to use the `AMI_GetWave()` function of an IBIS-AMI model, which doesn't provide one!")
    else:
        assert param_cfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"]), RuntimeError(
            "You've requested to use the `AMI_Init()` function of an IBIS-AMI model, which doesn't return an impulse response!")

    # Load and initialize the model.
    model_init = AMIModelInitializer(param_cfg.input_ami_params, info_params=param_cfg.info_ami_params)
    model_init.sample_interval = ts  # Must be set, before 'channel_response'!
    model_init.channel_response = chnl_h / ts
    model_init.bit_time = ui
    model = AMIModel(dll_fname)
    model.initialize(model_init)
    params_out = model.ami_params_out
    msg = "\n".join([  # Python equivalent of Haskell's `unlines()`.
        f"Input parameters: {model.ami_params_in.decode('utf-8')}",
        f"Output parameters: {params_out.decode('utf-8')}",
        f"Message: {model.msg.decode('utf-8')}"])

    # Capture model's responses.
    resps = model.get_responses()
    if use_getwave:
        h = resps["imp_resp_getw"]
        out_h = resps["out_resp_getw"]
    else:
        h = resps["imp_resp_init"]
        out_h = resps["out_resp_init"]

    # Generate model's output.
    if use_getwave:
        y, clks = model.getWave(x, bits_per_call=bits_per_call)
    else:
        y = convolve(x, out_h)[:len(x)]
        clks = None

    return (y, clks, h, out_h, msg)
