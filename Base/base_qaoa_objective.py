#from .fur import choose_simulator, choose_simulator_xyring, QAOAFastSimulatorBase
#from Parameter_utility import QAOAParameterization





def _get_qiskit_objective(
         parameterized_circuit,
         precomputed_optimal_bitstrings=None,
         objective: str = "expectation",
         terms=None,
         parameterization: str | QAOAParameterization = "theta",
         mixer: str = "x",
         optimization_type = "min"):
    #N = parameterized_circuit.num_qubits
    pass

def get_qaoa_objective():
    pass