"""
General IBIS-AMI utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>  
Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

A partial extraction of the old `pybert/utility.py`, as part of a refactoring.
"""

from numpy import array, convolve  # type: ignore

from pyibisami.ami.model import AMIModel, AMIModelInitializer
from pyibisami.ami.parser import AMIParamConfigurator

from ..common import Rvec


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def run_ami_model(dll_fname: str, param_cfg: AMIParamConfigurator, use_getwave: bool,
                  ui: float, ts: float, chnl_h: Rvec, x: Rvec, bits_per_call: int = 0  # noqa: F405
                  ) -> tuple[Rvec, Rvec, Rvec, Rvec, str, list[str]]:  # noqa: F405
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
        bits_per_call: Number of bits per call of ``GetWave()``.
            Default: 0 (Means "Use existing value.")

    Returns:
        y, clks, h, out_h, msg, params_out: A tuple consisting of:
            - the model output convolved w/ any channel impulse response given in `chnl_h`,
            - the model determined sampling instants (a.k.a. - "clock times"), if appropriate,
            - the model's impulse response (V/sample),
            - the impulse response of the model concatenated w/ the given channel (V/sample),
            - input parameters, and any message returned by the model's AMI_Init() function, and
            - any output parameters from GetWave() if apropos.

    Raises:
        IOError: if the given file name cannot be found/opened.
        RuntimeError: if the given model doesn't support the requested mode.
    """

    # Validate the model against the requested use mode.
    if use_getwave:
        if not param_cfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"]):
            raise RuntimeError(
                "You've requested to use the `AMI_GetWave()` function of an IBIS-AMI model, which doesn't provide one!")
    else:
        if not param_cfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"]):
            raise RuntimeError(
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
    resps = model.get_responses(bits_per_call=40)
    if use_getwave:
        h = resps["imp_resp_getw"]
        out_h = resps["out_resp_getw"][1]
    else:
        h = resps["imp_resp_init"]
        out_h = resps["out_resp_init"][1]

    # Generate model's output.
    if use_getwave:
        y, clks, params_out = model.getWave(x, bits_per_call=bits_per_call)
        return (y, clks, h, out_h, msg, list(map(lambda p: p.decode('utf-8'), params_out)))
    try:
        y = convolve(x, out_h)[:len(x)]
    except Exception:
        print(f"x.shape: {x.shape}")
        print(f"out_h shapes: {[h.shape for h in out_h]}")
        raise
    clks = array([])
    return (y, clks, h, out_h, msg, [])
