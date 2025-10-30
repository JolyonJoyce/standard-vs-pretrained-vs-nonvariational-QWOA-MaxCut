import numpy as np
# import matplotlib.pyplot as plt
# import random
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from standard_QAOAmc import PQC_sim_setup
import QAOAmc_core_components as core

def ramp_params(Q, p, alpha, beta, gamma, t):
    gammas = alpha * np.linspace(beta * gamma, gamma, p)/np.std(Q)
    ts = np.linspace(t, beta * t, p)
    return gammas, ts

#map given parameters to quantum circuit
def map_params(Q, alpha, beta, gamma, t, qc, gamma_parameters, t_parameters, p):
    gammas, ts = ramp_params(Q, p, alpha, beta, gamma, t)
    params_dict = {}
    for i in range(p):
        params_dict[gamma_parameters[i]] = gammas[i]
        params_dict[t_parameters[i]] = ts[i]
    qc = qc.assign_parameters(params_dict)
    return qc

# classical optimization
def sim_and_optimize(backend, Q, qc, G, gamma_parameters, t_parameters, alpha, beta_init, gamma_init, t_init, p, return_max_prob=False, optimize=False, method='L-BFGS-B', bounds=None, options = {'maxiter': 100}): 
    
    n = qc.num_qubits
    def cost(params):
        # Unpack the parameters
        beta_guess = params[0]
        gamma_guess = params[1]
        t_guess = params[2]
        
        # Map parameters to quantum circuit
        qc_opt = map_params(Q, alpha, beta_guess, gamma_guess, t_guess, qc, gamma_parameters, t_parameters, p)
        
        # Use the backend to calculate the expectation value
        statevector = backend.run(qc_opt).result().get_statevector()
        
        expectation_value = Q @ (np.abs(statevector)**2)
        print(-expectation_value) # to track minimization trajectory 
        return -expectation_value  # Negative for minimization
    
    '''
    derivativeM = []
    for i in range(n):
        dM = QuantumCircuit(n)
        dM.x(i)
        derivativeM += [dM]

    
    # gradient vector calculated according to the procedure outlined by Jones et al 2024, reference 144 of the associated thesis
    def gradients(params):
        # Unpack the parameters
        beta_guess = params[0]
        gamma_guess = params[1]
        t_guess = params[2]
        
        # Map parameters to quantum circuit
        qc_guess = map_params(Q, alpha, beta_guess, gamma_guess, t_guess, qc, gamma_parameters, t_parameters, p)
        
        # Use the backend to calculate the expectation value
        sv = backend.run(qc_guess).result().get_statevector()
        phi = Statevector(sv.data)
        lm = sv.data*Q
        lam = Statevector(lm)
        
        # Compute gradients
        d_beta = 0
        d_gamma = 0
        d_t = 0
        
        return np.array([d_beta, d_gamma, d_t])
        
        """ standard approach: optimization of all parameters
        # Compute gradients
        d_ts = np.zeros(p)
        d_gammas = np.zeros(p)   
        for i in range(p-1,-1,-1): #(start(incl), stop(excl), increment): starts at (p-1)th layer and stops at 0th layer 
            d_ts[i] = 2*sum(1.0j*lam.conjugate().data@phi.evolve(derivativeM[i]).data for i in range(n)).real

            udag = core.reverse_mixing_operator(n,ts_opt[i])
            phi = phi.evolve(udag)

            lam = lam.evolve(udag)

            d_gammas[i] = 2*(1j*lam.conjugate().data@(Q*phi.data)).real

            udag = core.reverse_phase_separator(n,G,gammas_opt[i])
            phi = phi.evolve(udag)

            lam = lam.evolve(udag)
        return np.concatenate((d_gammas, d_ts))
        """
    '''

    if optimize:
        # Combine initial parameters into one array
        initial_params = np.array([beta_init, gamma_init, t_init])
        
        # Default bounds are not provided
        if bounds is None:
            bounds = [(0,1),(0,2*np.pi),(0,2*np.pi)]  # Assuming ranges [0, 2π] for all parameters
        
        # Minimize the negative expectation value
        result = minimize(cost, initial_params, bounds=bounds, method=method, options=options)
    
        num_iters = result.nit
        
        # Extract the best found params
        best_params = result.x
        best_beta = best_params[0]
        best_gamma = best_params[1]
        best_t = best_params[2]
            
        # The maximum expectation value is the negative of the minimum found
        max_expectn_value = -result.fun
        approxn_ratio = max_expectn_value/max(Q)
            
        # Get the probablility of measuring a maximum cut
        qc = map_params(Q, alpha, best_beta, best_gamma, best_t, qc, gamma_parameters, t_parameters, p)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        prob_max = sum(probs[np.where(Q == max(Q))[0]]) #probability of measuring a maximum cut = sum of probability of each maximum cut
        
        return result, num_iters, approxn_ratio, prob_max, best_beta, best_gamma, best_t
    
    else:
        # Calculate expectation with given parameters
        qc = map_params(Q, alpha, best_beta, best_gamma, best_t, qc, gamma_parameters, t_parameters, p)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        
        if return_max_prob:
            return Q @ probs, sum(probs[np.where(Q == max(Q))[0]]),probs
        else:
            return Q @ probs

# run the full VQA (randomly initialized parameters)
def run(backend, Q, G, p, q, alpha=False, beta=False, gamma=False, t=False):
    # set up
    #- cut sizes
    Q = core.qualities(G)
    # print("CUT SIZE: max: ", max(Q), " mean: ", np.mean(Q), " med: ", np.median(Q))
    
    #- prepare equal superposition and label parameters in PQC
    qc, gamma_parameters, t_parameters = PQC_sim_setup(backend, G, p)
    
    #- initialize parameters (can accept parameters if you wish to train in chunks of iterations)
    if type(alpha) == bool:
        alpha = 1
    if type(beta) == bool:
        beta = 0.4 
        #beta = np.random.normal(1., 0.5)
    if type(gamma) == bool:
        gamma = 1
        #gamma = np.random.normal(1., 0.5)
    if type(t) == bool:
        t = 0.3
        #t = np.random.normal(1., 0.5)
    
    # simulation
    scipy_output, num_iters, approxn_ratio, prob_max, best_beta, best_gamma, best_t = sim_and_optimize(backend, Q, qc, G, gamma_parameters, t_parameters, alpha, beta, gamma, t, p, optimize=True, options={"maxfun": 50000, "maxiter": q, 'eps':1e-8, 'ftol':1e-8, 'gtol':1e-5, 'maxcor':40})

    return scipy_output, num_iters, approxn_ratio, prob_max, best_beta, best_gamma, best_t
