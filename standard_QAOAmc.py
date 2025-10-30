import numpy as np
# import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import QAOAmc_core_components as core

# set up for Qiskit simulation of the PQC of given depth p, for an iteration of classical optimization
def PQC_sim_setup(backend, G, p): 
    n = len(G)
    qc = core.initialize_qubits(n)
    gamma_parameters = [Parameter(f"gamma_{i}") for i in range(p)] #to assign names to the PQC's parameters
    t_parameters = [Parameter(f"t_{i}") for i in range(p)] #sim.
    for i in range(p):
        core.apply_phase_separator(qc, G, gamma_parameters[i])
        core.apply_mixing_operator(qc, t_parameters[i])
    qc.save_statevector()
    qc = transpile(qc, backend=backend)
    return qc, gamma_parameters, t_parameters

'''this is not used in simulating the VQA
#run PQC with given parameters and return modulus squared of amplified state vector
def amplified_state(Q, alpha, beta, gamma, ts, qc, gamma_parameters, t_parameters, p):
    qc = map_params(Q, alpha, beta, gamma, t, qc, gamma_parameters, t_parameters, p)
    return np.abs(backend.run(qc).result().get_statevector())**2
'''

def map_params(gammas, ts, qc, gamma_parameters, t_parameters, p):
    params_dict = {}
    for i in range(p):
        params_dict[gamma_parameters[i]] = gammas[i]
        params_dict[t_parameters[i]] = ts[i]
    qc = qc.assign_parameters(params_dict)
    return qc

def sim_and_optimize(backend, Q, qc, G, gamma_parameters, t_parameters, gammas, ts, p, return_max_prob=False, optimize=False, method='L-BFGS-B', bounds=None, options = {'maxiter': 100} ):
    n = qc.num_qubits
    
    # As currently formulated, this approach requires running the simulation TWICE for each iteration: once to get the expectation value of the cut size (cost(params)) and once to get the gradients of each parameter (gradients(params))
    # This must be slowing it down. It would be better to run once at each iteration and feed the expectation value and gradients to the minimizer.
    
    def cost(params): #how good are the found parameters? lower returned value = higher cut size = better :)
        # Unpack the (optimized) parameters
        gammas_opt = params[:p]
        ts_opt = params[p:]
        # Map parameters to quantum circuit
        qc_opt = map_params(gammas_opt, ts_opt, qc, gamma_parameters, t_parameters, p)
        # Use the backend to calculate the expectation value
        statevector = backend.run(qc_opt).result().get_statevector()
        
        expectation_value = Q @ (np.abs(statevector)**2) # matrix product of list of qualities Q with probability vector = expectation value of cut size
        print(-expectation_value) # to track minimization trajectory 
        return -expectation_value  # Negative for minimization
    
    derivativeM = []
    for i in range(n):
        dM=QuantumCircuit(n)
        dM.x(i)
        derivativeM+= [dM]

    # gradients are calculated according to the procedure outlined by Jones et al 2024, reference 144 of the associated thesis
    def gradients(params):
        # Unpack the parameters
        gammas_opt = params[:p]
        ts_opt = params[p:]
        
        # Map parameters to quantum circuit
        qc_opt = map_params(gammas_opt, ts_opt, qc, gamma_parameters, t_parameters, p)
        
        # Use the backend to calculate the expectation value
        sv = backend.run(qc_opt).result().get_statevector()
        phi = Statevector(sv.data)
        lm=sv.data*Q
        lam = Statevector(lm)
        
        # compute gradients
        d_ts = np.zeros(p)
        d_gammas = np.zeros(p)   
        
        for i in range(p-1,-1,-1): #(start(incl), stop(excl), increment): starts at (p-1)th layer and stops at 0th layer 

            d_ts[i] = 2*sum(1.0j*lam.conjugate().data@phi.evolve(derivativeM[i]).data for i in range(n)).real

            udag = core.reverse_mixing_operator(n,ts_opt[i])
            phi=phi.evolve(udag)

            lam=lam.evolve(udag)

            d_gammas[i] = 2*(1j*lam.conjugate().data@(Q*phi.data)).real

            udag = core.reverse_phase_separator(n,G,gammas_opt[i])
            phi=phi.evolve(udag)

            lam=lam.evolve(udag)
        
        return np.concatenate((d_gammas, d_ts))

    if optimize:
        # Combine initial gamma and t parameters into one array
        initial_params = np.concatenate((gammas, ts))
        
        # Default bounds if not provided
        if bounds is None:
            bounds = [(0, 2*np.pi)] * (2*p)  # Assuming ranges [0, 2π] for all parameters
        
        # Minimize the negative expectation value
        result = minimize(cost, initial_params, bounds=bounds, method=method,jac=gradients,options=options)

        #scipy_output = result
        num_iters = result.nit
        
        # Extract the best found params
        best_params = result.x
        best_gammas = best_params[:p]
        best_ts = best_params[p:]
            
        # The maximum expectation value is the negative of the minimum found
        max_expectation_value = -result.fun
        approxn_ratio = max_expectation_value/max(Q)
            
        # Get the probablility of measuring a maximum cut
        qc = map_params(best_gammas, best_ts, qc, gamma_parameters, t_parameters, p)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        prob_max = sum(probs[np.where(Q == max(Q))[0]]) #probability of measuring a maximum cut = sum of probability of each maximum cut
        

        return result, num_iters, approxn_ratio, prob_max, best_gammas, best_ts
    
    else:
        # Calculate expectation with given parameters
        qc = map_params(gammas, ts, qc, gamma_parameters, t_parameters, p)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        if return_max_prob:
            return Q @ probs, sum(probs[np.where(Q == max(Q))[0]]),probs
        else:
            return Q @ probs
        
# run the full VQA (randomly initialized parameters)
def run(backend, Q, G, p, q, gammas=False, ts=False):    
    # set up    
    #- prepare equal superposition and label parameters in PQC
    qc, gamma_parameters, t_parameters = PQC_sim_setup(backend, G, p)
    
    #- initialize parameters (can accept parameters if you wish to train in chunks of iterations)
    if type(ts) == bool:
        ts = np.random.normal(1., 0.5, size=(p,))
    else:
        ts = np.array(ts)
    if type(gammas) == bool:
        gammas = np.random.normal(1., 0.5, size=(p,))
    else:
        gammas = np.array(gammas)
    
    # simulation
    scipy_output, num_iters, approxn_ratio, prob_max, best_gammas, best_ts = sim_and_optimize(backend, Q, qc, G, gamma_parameters, t_parameters, gammas, ts, p, optimize=True, options={"maxfun": 50000, "maxiter": q})

    return scipy_output, num_iters, approxn_ratio, prob_max, best_gammas, best_ts