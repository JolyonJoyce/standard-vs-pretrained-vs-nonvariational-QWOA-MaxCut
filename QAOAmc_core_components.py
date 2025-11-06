import numpy as np
# import matplotlib.pyplot as plt
# import random
import networkx as nx
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter

# I don't know what this is for...
def unpackbits(x, num_bits, asbool=True):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits-1,-1,-1, dtype=x.dtype).reshape([1, num_bits])
    if asbool:
        return (x & mask).astype(bool).reshape(xshape + [num_bits])
    else:
        return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])
# nor this...
def to_int(x):
    # Reverse the order of the bits to convert back to normal binary representation
    return int(''.join([str(i) for i in x[::-1]]), 2)
# nor this...
def to_binary(x, bits):
    # Convert the integer to binary, fill with leading zeros, and reverse the order
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)][::-1])

'''===== Problem Preparation ====='''

# function to generate weighted/unweighted simple (no loops, undirected) graph G on n vertices of various types (path, cycle, complete, 3-regular, Erdős-Rényi random)
# - output is weight matrix (if unweighted, weights are 1, so weight matrix is simply adjacency matrix)

def generate_graph(n=4, G_type="ER", weighted=False, edge_prob=0.3, seed=None):
    '''
    Parameters:
    - n (int): order of graph, i.e. number of vertices;
    - edge_prob (float): if G_type="ER" then every possible edge has an equal probability edge_prob of existing;
    - seed (int or bool): seed for random number generator;
    - weighted (bool): whether or not to assign random weights to edges;
    - G_type (str): type of graph to generate:
        - "path" : a path graph of order n.
        - "cycle" : a cycle graph of order n.
        - "complete" : the complete graph of order n (unweighted only). # does weighted not work? why?
        - "3-reg" : a 3-regular graph.
        - "pseudo-3-reg" : different method of generating 3-regular graphs with very small non-zero probability of generating a non-3-regular graph.
        - "ER" : a graph generated using the Erdős-Rényi random graph model (every possible edge has an equal probability edge_prob of existing).

    Returns:
    - G : Adjacency matrix of the generated graph.
    '''
    # set the random seed for reproducibility
    if isinstance(seed, int):
        np.random.seed(seed)
    elif seed is True:
        np.random.seed(42)

    if G_type == "path":
        G = np.zeros((n, n))
        for i in range(n - 1):
            G[i, i + 1] = 1 - np.random.random() if weighted else 1 #subsequently indexed vertices are adjacent
            G[i + 1, i] = G[i, i + 1] #reflect across diagonal
        if not weighted:
            G = G.astype(int) #ensure unweighted graphs have integer entries

    elif G_type == "cycle":
        G = np.zeros((n, n))
        for i in range(n - 1):
            G[i, i + 1] = 1 - np.random.random() if weighted else 1 #subsequently indexed vertices are adjacent
            G[i + 1, i] = G[i, i + 1] #reflect across diagonal
        G[n-1, 0] = 1 - np.random.random() if weighted else 1 #first and last indexed vertices are adjacent
        G[0, n-1] = G[n-1, 0] #reflect across diagonal
        if not weighted:
            G = G.astype(int) #ensure unweighted graphs have integer entries

    elif G_type == "complete":
        #if weighted:
        #    raise ValueError("Complete graphs cannot be weighted. Please set weighted=False.") # why can completed graphs not be weighted ?
        # Create a complete graph without self-loops
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n): #for elements above diagonal...
                G[i, j] = 1 - np.random.random() if weighted else 1 #(every pair of vertices is adjacent)
                G[j, i] = G[i, j] #reflect accross diagonal
        if not weighted:
            G = G.astype(int) #ensure unweighted graphs have integer entries

    elif G_type == "3-reg":
        if n % 2 != 0:
            raise ValueError("3-regular graphs only exist for even numbers of vertices.")
        G_nx = nx.random_regular_graph(3, n, seed=seed if isinstance(seed, int) else None)
        G = np.array(nx.to_numpy_array(G_nx))
        if weighted:
            # Assign random weights to existing edges
            weight_matrix = np.ones((n,n)) - np.random.random((n, n)) #ensure weight coefficients are non-zero
            G = G * weight_matrix
            for i in range(n):
                for j in range(i + 1, n):
                    G[j, i] = G[i, j] #ensure symmetry
        else:
            G = G.astype(int) #ensure unweighted graphs have integer entries
         
    elif G_type == "pseudo-3-reg": #this was the method used in Matthaus' thesis code; generating weighted graphs of this type with seed = 0 through 199 will produce the graphs used in Matthaus' thesis
        if n * 3 % 2 != 0:
            raise ValueError("3-reg graphs are only possible for an even number of nodes.")
        G_nx = nx.random_regular_graph(3, n, seed=seed if isinstance(seed, int) else None)
        G = np.array(nx.to_numpy_array(G_nx))
        if weighted:
            # Assign random weights to existing edges
            weight_matrix = np.random.random((n, n))
            G = G * weight_matrix
        else:
            G = G.astype(int) #ensure unweighted graphs have integer entries
            
    elif G_type == "ER":
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < edge_prob:
                    G[i, j] = np.random.random() if weighted else 1
                    G[j, i] = G[i, j]
        if weighted == False:
            G = G.astype(int) #ensure unweighted graphs have integer entries

    else:
        raise ValueError(f"Unknown graph type: {G_type}")

    return G


## function to produce QUBO representation Q of graph G
def QUBO(G):
    return -(G - np.diag(G.sum(axis=1))) # (where does this come from?)

## function to produce the size of a cut as a binary string x of a graph with QUBO representation Q
def quality(vec_x, qubo_G):
    return vec_x.T @ qubo_G @ vec_x # (@ is mATrix product; vec_x.T is transpose of vec_x)

# compute the quality vector; N.B. quality vector is symmetric about centre point
def qualities(G):
    n = len(G)
    qubo_G = QUBO(G)
    N = 2**n
    Q = np.zeros(N)
    all_binary_arrays = unpackbits(np.arange(N//2), n, asbool=True)
    
    for i in range(N//2):
        Q[i] = quality(all_binary_arrays[i], qubo_G)
    
    Q[N//2:] = (Q[:N//2])[::-1]
    
    return Q #= quality vector of graph G

'''===== Circuit Components ====='''
# Hadamard (equal superposition)
def initialize_qubits(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    return qc

# hypercube mixer (NOT gate on each qubit) for time t
def apply_mixing_operator(qc, t):
    for qubit in range(qc.num_qubits):
        qc.rx(2*t, qubit)

# phase separation based on cut values of graph G, scaled by coefficient gamma
def apply_phase_separator(qc, G, gamma):
    n = qc.num_qubits
    for i in range(n-1):
        for j in range(i+1,n):
            if G[i, j] != 0:
                qc.cx(n-1-i, n-1-j)
                qc.p(-gamma*G[i, j], n-1-j)
                qc.cx(n-1-i, n-1-j)

# reverse operators...
def reverse_mixing_operator(n,t):
    qc = QuantumCircuit(n)
    for qubit in range(qc.num_qubits):
        qc.rx(-2*t, qubit)
    return qc

def reverse_phase_separator(n,G, gamma):
    qc = QuantumCircuit(n)
    for i in range(n-1):
        for j in range(i+1,n):
            if G[i, j] != 0:
                qc.cx(n-1-i, n-1-j)
                qc.p(gamma*G[i, j], n-1-j)
                qc.cx(n-1-i, n-1-j)
    return qc
