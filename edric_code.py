from scipy.optimize import minimize
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import networkx as nx
import QAOAmc_core_components as core

NUM_GRAPHS_PER_SIZE = 200

def initialize_qubits(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    return qc

def apply_mixing_operator(qc, t):
    for qubit in range(qc.num_qubits):
        qc.rx(2*t, qubit)

def apply_phase_separator(qc, G, Q, gamma):
    n = qc.num_qubits
    for i in range(n-1):
        for j in range(i+1,n):
            if G[i, j] != 0:
                qc.cx(n-1-i, n-1-j)
                qc.p(-gamma*G[i, j], n-1-j)
                qc.cx(n-1-i, n-1-j)

def qiskit_qwoa(G, Q, alpha, beta, p, gamma, t):
    n = len(G)
    gammas = alpha*np.linspace(beta*gamma, gamma, p)/np.std(Q)
    times = np.linspace(t, beta*t, p)
    qc = initialize_qubits(n)
    for i in range(p):
        apply_phase_separator(qc, G, Q, gammas[i])
        apply_mixing_operator(qc, times[i])
    qc.save_statevector()
    backend = AerSimulator(method='statevector')
    job = backend.run(qc)
    result = job.result()
    statevector = result.get_statevector()
    probabilities = np.abs(statevector)**2 #operations here (modulus squared of complex amplitudes) are element-wise, hence result is probability vector
    return probabilities

'''
def expectation_value(x, G, Q, alpha, beta, p):
    probabilities = qiskit_qwoa(G, Q, alpha, beta, p, x[0], x[1])
    expectation = np.sum(probabilities * Q)
    return -expectation  # Minimise the negative expectation for maximisation
'''
def expectation_value_optimize_for_beta(x, G, Q, alpha, p):
    probabilities = qiskit_qwoa(G, Q, alpha, x[2], p, x[0], x[1])
    expectation = np.sum(probabilities * Q)
    return -expectation  # Minimise the negative expectation for maximisation

'''
def optimal_probability(x, G, Q, alpha, beta, p, opt_mask):
    probabilities = qiskit_qwoa(G, Q, alpha, beta, p, x[0], x[1])
    opt_prob = np.sum(probabilities[opt_mask])
    return -opt_prob # Minimise the negative optimal probability for maximisation
'''
def optimal_probability_optimize_for_beta(x, G, Q, alpha, p, opt_mask):
    probabilities = qiskit_qwoa(G, Q, alpha, x[2], p, x[0], x[1])
    opt_prob = np.sum(probabilities[opt_mask])
    return -opt_prob # Minimise the negative optimal probability for maximisation`  

'''
def tune(G, Q, alpha, beta, p):
    if not opt_mask.size:
        x = minimize(expectation_value, [1, 0.1], args=(G, Q, alpha, beta, p)).x
    return x[0], x[1]
'''
# To tune for expectation value, do not pass opt_mask
def tune_optimize_for_beta(G, Q, alpha, p, initial_guess=[1, 0.3, 0.4], opt_mask=[]):
    method='L-BFGS-B'
    options={'eps':1e-8, 'ftol':1e-8, 'gtol':1e-5, 'maxcor':40}
    if len(opt_mask) == 0:
        x = minimize(expectation_value_optimize_for_beta, initial_guess, bounds=[(0,5),(0,0.7),(0,0.5)], args=(G, Q, alpha, p), method=method, options=options).x
    else:
        x = minimize(optimal_probability_optimize_for_beta, initial_guess, bounds=[(0,5),(0,0.7),(0,0.5)], args=(G, Q, alpha, p, opt_mask), method=method, options=options).x
    return x[0], x[1], x[2]

def simulate(G, alpha, p, graphs, n, initial_guess=[1, 0.3, 0.4], tune_for_optimal_probability=False):
    
    results = np.zeros((NUM_GRAPHS_PER_SIZE, 4))
    total_opt_prob = 0
    total_gamma = 0
    total_t = 0
    total_beta = 0
    
    for i in range(NUM_GRAPHS_PER_SIZE):
        G = graphs[i]
        Q = get_quality_vector(G)
        opt_mask = Q == Q.max()
        N = len(Q)
        
        if tune_for_optimal_probability:
            optimised_gamma, optimised_t, optimised_beta = tune_optimize_for_beta(G, Q, alpha, p, initial_guess, opt_mask)
        else:
            optimised_gamma, optimised_t, optimised_beta = tune_optimize_for_beta(G, Q, alpha, p, initial_guess)
        
        probs = qiskit_qwoa(G, Q, alpha, optimised_beta, p, optimised_gamma, optimised_t)
        opt_prob = probs[opt_mask].sum()
        
        results[i] = [optimised_gamma, optimised_t, optimised_beta, opt_prob]
        
        total_opt_prob += opt_prob
        total_gamma += optimised_gamma
        total_t += optimised_t
        total_beta += optimised_beta
    
    results_numpy_array = np.array(results)
    np.save(f'results/results_n={n}_p={p}.npy', results_numpy_array)
    
    avg_opt_prob = total_opt_prob / NUM_GRAPHS_PER_SIZE
    avg_gamma = total_gamma / NUM_GRAPHS_PER_SIZE
    avg_t = total_t / NUM_GRAPHS_PER_SIZE
    avg_beta = total_beta / NUM_GRAPHS_PER_SIZE
    return avg_opt_prob, avg_gamma, avg_t, avg_beta

'''
def report(G, Q, alpha, beta, p, gamma, t, plot=False):
    N = len(Q)
    opt_mask = Q == Q.max()
    AR = Q / Q.max()
    mask = AR > 0.99
    opt_count = opt_mask.sum()
    AR_count = mask.sum()
    probs = qiskit_qwoa(G, Q, alpha, beta, p, gamma, t)
    opt_prob = probs[opt_mask].sum()
    AR_prob = probs[mask].sum()
    amp = opt_prob * N / opt_count
    
    print(f'p={p}, gamma={gamma:.6f}, t={t:.6f}')
    print(f'Global optimal probability = {opt_prob:.6f}, amplification = {amp:.4f}')
    print(f'Probability Within 1% AR: {AR_prob:.6f}, number of solutions: {AR_count}')
    
    if plot:
        plt.hist(AR, weights=np.ones(N) / N, bins=200, alpha=0.8, label='unamplified')
        plt.hist(AR, weights=probs, bins=200, alpha=0.8, label='amplified')
        plt.xlabel('Approximation ratio')
        plt.ylabel('Measurement probability')
        plt.legend()
        plt.show()
'''

alpha = 1
p = 1
initial_guess = [1, 0.3, 0.4]
for n in range(10, 10001):
    graphs = np.zeros((NUM_GRAPHS_PER_SIZE, n, n))
    
    for i in range(NUM_GRAPHS_PER_SIZE):
        
        # Generate a connected graph
        while True:
            G = random_graph(n)
            nx_graph = nx.from_numpy_array(G)
            if nx.is_connected(nx_graph):
                break
        
        graphs[i] = G
    
    np.save(f'results/graphs_n={n}.npy', graphs)
    
    p_has_increased_for_this_n = False
    avg_opt_prob, avg_gamma, avg_t, avg_beta = simulate(G, alpha, p, graphs, n, initial_guess)
    while avg_opt_prob < 0.1:
        p += 1
        p_has_increased_for_this_n = True
        avg_opt_prob, avg_gamma, avg_t, avg_beta = simulate(G, alpha, p, graphs, n, initial_guess)
    
    if not p_has_increased_for_this_n:
        p -= 1
        avg_opt_prob, avg_gamma, avg_t, avg_beta = simulate(G, alpha, p, graphs, n, initial_guess)
        p += 1
    
    initial_guess = [avg_gamma, avg_t, avg_beta]
    