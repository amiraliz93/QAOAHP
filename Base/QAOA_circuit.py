import networkx  as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from typing import Sequence
import matplotlib.pyplot as plt

def append_zz_term(qc, q1, q2, gamma):
    qc.rzz(-gamma / 2, q1, q2) 

def append_maxcut_cost_operator_circuit(qc, G, gamma):
    for i, j in G.edges():
        if nx.is_weighted(G):
            append_zz_term(qc,i,j,gamma *G[i][j]['weight'])
        else:
            append_zz_term(qc, i, j, gamma)

def append_x_term(qc, q1, beta):
    qc.rx(2*beta, q1)

def append_mixer_operator_circuit(qc, G, beta):
    for n in G.nodes():
        append_x_term(qc, n, beta)

def get_qaoa_circuit(G: nx.Graph, gammas: Sequence, betas: Sequence, save_statvector:bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None ):
    """Generate a circuit for weighted MaxCut on Graph g
    Parameters
    -------------
    G : networkx.Graph
    beta : list-like
    gamma : list-like
    save_statevector : bool, default True
    qr : qiskit.QuantumRegister, default None
    Registers to use for the circuit.
    Useful when one has to compose circuits in a complicated way
        By default, G.number_of_nodes() registers are used
    cr : qiskit.ClassicalRegister, default None
        Classical registers, useful if measuring
        By default, no classical registers are added
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing QAOA
    """
    assert len(betas) == len(gammas)
    p = len(betas)  # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes
    if qr is not None:
        assert qr.size >=N
    else:
        qr = QuantumRegister(N)
    if cr is not None:
        qc = QuantumCircuit(qr, cr) 
    else:
        qc = QuantumCircuit(qr)
    
    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        append_maxcut_cost_operator_circuit(qc, G, gammas[i])
        append_mixer_operator_circuit(qc, G, betas[i])
    if save_statvector:
        qc.save_statevector()
    return qc

def get_parameterized_qaoa_circuit(G: nx.Graph, p: int, save_statevector: bool = True, qr: QuantumRegister = None, cr: ClassicalRegister = None, return_parameter_vectors: bool = False):
    """Generates a parameterized circuit for weighted MaxCut on graph G.
    This version is recommended for long circuits

    Parameters

    Args:
        G (nx.Graph): networkx.Graph
        p (int): int :  Number of QAOA layers (number of parameters will be 2*p)
        save_statevector (bool, optional): add save state instruction to the end of the circuit. Defaults to True.
        qr (QuantumRegister, optional): Registers to use for the circuit. Useful when one has to compose circuits in a complicated way . 
         By default, G.number_of_nodes() registers are use
        cr (ClassicalRegister, optional): Classical registers, useful if measuring
        By default, no classical registers are adde
             Defaults to None.
        return_parameter_vectors (bool, optional): Return ParameterVector for betas and gammas
        . Defaults to False.
        
    Returns
    -------
    qc : qiskit.QuantumCircuit
        Parameterized quantum circuit implementing QAOA
        Parameters are two ParameterVector sorted alphabetically
        (beta first, then gamma). To bind:
        qc.bind_parameters(np.hstack([angles['beta'], angles['gamma']]))
    """
    N =G.number_of_nodes()
    if qr is not None:
        assert qr.size >=N
    else:
        qr = QuantumRegister(N)
    if cr is not None:
        qc = QuantumCircuit(qr, cr)
    else:
        qc = QuantumCircuit(qr)
    
    beta = ParameterVector("beta", p)
    gamma = ParameterVector("gamma", p)

    # first , apply a layer of hadamard
    qc.h(range(N))
    # second apply p operator
    for i in range(p):
        append_maxcut_cost_operator_circuit(qc, G, gamma[i])
        append_mixer_operator_circuit(qc, G, beta[i])

    if save_statevector:
        qc.save_statevector()
    if return_parameter_vectors:
        return qc, beta, gamma
    else:
        return qc


#G = nx.erdos_renyi_graph(4,.8)
#nx.draw(G, with_labels=True)
#plt.show()
#qc = get_qaoa_circuit(G)
