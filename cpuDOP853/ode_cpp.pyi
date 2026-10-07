# ode_cpp.pyi
import numpy as np
from typing import List, Tuple, Union
import numpy.typing as npt

def run_simulation(
    z0: float, 
    v0: float, 
    breakpoints: Union[List[float], npt.NDArray[np.float64]], 
    k: float, 
    gEf: float, 
    B: float,
    zEq: float,
    Lambda: float,
    val0: float = 0.0, 
    val1: float = 2.5, 
    base_A: float = 2.5,
    base_phi: float = 0.0,
    dA: float = 0.0,
    omega: float = 0.0,
    max_peaks: int = 100,
    trackerStartTime: float = 0.0,
    simMode: int = 0,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    initialStep: float = 1e-4
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], 
           npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64],
           Tuple[bool, int, int, int]]:
    """
    Run DOP853 ODE integration through defined sub-intervals.
    
    **Parameters:**
    -----------
    z0 : float
        Initial position.
    v0 : float
        Initial velocity.
    breakpoints : List[float]
        Strictly increasing list of sub-intervals end times.
        breakpoints[0] is the initial time.
    k : float
        Wavenumber parameter.
    gEf : float
        Gravity parameter.
    B : float
        Damping coefficient.
    zEq : float
        Equilibrium position (for checking if the object was lost).
    Lambda : float
        Acoustic wavelength (for checking if the object was lost).
    val0 : float, optional
        Amplitude (if simMode == 0) or Phase (if simMode == 1) during even sub-intervals. Defaults to 0.
        The parameter is ignored if simMode == 2.
    val1 : float, optional
        Amplitude (if simMode == 0) or Phase (if simMode == 1) during odd sub-intervals. Defaults to 2.5.
        The parameter is ignored if simMode == 2.
    base_A : float, optional
        Amplitude used if (simMode == 1 or simMode == 2). Defaults to 2.5.
    base_phi : float, optional
        Phase used if (simMode == 0 or simMode == 2). Defaults to 0.
    dA : float, optional
        Factor of the Amplitude Modulation, only used for simMode == 2. Defaults to 0.
    omega : float, optional
        Angular Frequency of the Amplitude Modulation, only used if simMode == 2. Defaults to 0.
    max_peaks : int, optional
        Max number of each kind of peak to track. Defaults to 100.
        If not positive, all peaks ocurrying after trackerStartTime are tracked.
    trackerStartTime : float, optional
        Time boundary at which the peaks should start to be tracked. Defaults to 0.
        Affect both the tracking for a fixed number of peaks (max_peaks > 0) and unlimited peaks (max_peaks <=0).
    simMode : int, optional
        The driving mode of the system (0 -> on/off [Amplitude], 1 -> jumps [Phase], 2-> modulation [Sine])
        For on/off, val0 and val1 represents the amplitudes, base_A is overrided and the phase is base_phi.
        For jumps, val0 and val1 represents the phases, base_phi is overrided and the amplitude is base_A.
        For modulation, val0 and val1 aren't used, the modulation is defined by omega and dA, the amplitude is base_A, the phase is base_phi.
    rtol : float
        Max relative tolerance (default = 1e-7).
    atol : float
        Max absolute tolerance (default = 1e-10).
    initialStep : float
        Initial step size for the integrator (default = 1e-4).
        
    Returns:
    --------
    Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], bool]
        A tuple containing seven arrays and a boolean flag: (time_history, z_history, v_history, t_max, z_max, t_min, z_min, (captured, simMode, nAccepted, nRejected) ).
    """
    ...