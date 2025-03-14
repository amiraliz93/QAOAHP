
import numba.cuda
from .qaoa_simulator_base import QAOAFastSimulatorBase
from .Simulators.python.QAOA_simulator import QAOAFURXSimulator, ParamType, CostsType, TermsType
from .Simulators.GPU.qaoa_simulatorbase import QAOAFURXSimulatorGPU
Simulators = {
    "x": {
       # "c": QAOAFURXSimulatorC,
        "python": QAOAFURXSimulator,
         "gpu": QAOAFURXSimulatorGPU,
        #"gpumpi": QAOAFURXSimulatorGPUMPI,
    }
}

def get_available_simulator_names(type: str = "x") -> list:
    """_summary_

    Args:
        type (str, optional): name of simulator. Defaults to "x".

    Returns:
        list: available simulator
    """
    family = Simulators.get(type, None)
    if family is None:
        raise ValueError(f'the simulator is not defined {type}')

    precedence = ["python",  "gpu" ] # "gpumpi", "c"

    check = [ numba.cuda.is_available] # mpi_available, , c_available
    available = []
    for i in range(len(check)):
        if precedence[i] not in family:
            continue
        if check[i]():
            available.append(precedence[i])
    
    available.append(precedence[-1])
    return available


def get_available_simulators(type: str = "x") -> list:
    """
    Return (uninitialized) classes of available simulators

    Parameters
    ----------
        type: type of QAOA mixer to simulate

    Returns
    -------
        List of available simulators
    """
    available_names = get_available_simulator_names(type=type)
    return [ [type][s] for s in available_names]


def choose_simulator(name="auto", **kwargs):
    if name != "auto":
        return Simulators["x"][name]
    #return Simulators["x"]['python']
     
    print(get_available_simulators("x"))
    return get_available_simulators("x")[0]