import numpy as np
import networkx as nx
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import QAOAmc_core_components as core

################# Pretraining QAOA EXAMPLE ############################

# the majority of functions are helper functions which are identical to the ones in the standard qaoa code
# difference is in the circuit design, which takes two graph structures and two sets of gammas
# this follwos the unitary defined by Equation 3.11 in the thesis

# set up for Qiskit simulation of the PQC of given depth p, for an iteration of classical optimization
def PQC_sim_setup(backend, G, Gprime, p):
    n = len(G)
    qc = core.initialize_qubits(n)
    gamma_parameters = [Parameter(f"gammaG_{i}") for i in range(p)]
    gamma_prime_parameters = [Parameter(f"gammaP_{i}") for i in range(p)]
    t_parameters = [Parameter(f"t_{i}") for i in range(p)]
    for i in range(p):
        core.apply_phase_separator(qc, G, gamma_parameters[i])
        core.apply_phase_separator(qc, Gprime, gamma_prime_parameters[i])
        core.apply_mixing_operator(qc, t_parameters[i])
    qc.save_statevector()
    qc = transpile(qc, backend=backend)
    return qc, gamma_parameters, gamma_prime_parameters, t_parameters

#map given parameters to quantum circuit
def map_params(gammas, gamma_primes, ts, qc, gamma_parameters, gamma_prime_parameters, t_parameters, p):
    params_dict = {}
    for i in range(p):
        params_dict[gamma_parameters[i]] = gammas[i]
        params_dict[gamma_prime_parameters[i]] = gamma_primes[i]
        params_dict[t_parameters[i]] = ts[i]
    qc = qc.assign_parameters(params_dict)
    return qc

# classical optimization
def sim_and_optimize(backend, Q, qc, G, Gprime, gamma_parameters, gamma_prime_parameters, t_parameters, gammas, gamma_primes, ts, p, return_max_prob=False,optimize=False, method='L-BFGS-B', bounds=None,options = {'maxiter': 100} ):
    
    # As currently formulated, this approach requires running the simulation TWICE for each iteration: once to get the expectation value of the cut size (cost(params)) and once to get the gradients of each parameter (gradients(params))
    # This must be slowing it down. It would be better to run once at each iteration and feed the expectation value and gradients to the minimizer.
    
    n = qc.num_qubits
    Qprime = core.qualities(Gprime)
    
    def cost(params):
        # Unpack the parameters
        gammas_guess = params[:p]
        gamma_primes_guess = params[p:2*p]
        ts_guess = params[2*p:]
        # Map parameters to quantum circuit
        qc_guess = map_params(gammas_guess,gamma_primes_guess, ts_guess, qc, gamma_parameters, gamma_prime_parameters, t_parameters, p)
        # Use the backend to calculate the expectation value
        statevector = backend.run(qc_guess).result().get_statevector()
        
        expectation_value = Q @ (np.abs(statevector)**2)
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
        gammas_guess = params[:p]
        gamma_primes_guess = params[p:2*p]
        ts_guess = params[2*p:]
        
        # Map parameters to quantum circuit
        qc_guess = map_params(gammas_guess,gamma_primes_guess, ts_guess, qc, gamma_parameters,gamma_prime_parameters, t_parameters, p)
        
        # Use the backend to calculate the expectation value
        sv = backend.run(qc_guess).result().get_statevector()
        phi = Statevector(sv.data)
        lm = sv.data*Q
        lam = Statevector(lm)
        
        # compute gradients
        d_ts = np.zeros(p)
        d_gammas = np.zeros(p)
        d_gamma_primes = np.zeros(p)
        
        for i in range(p-1,-1,-1): #(start(incl), stop(excl), increment): starts at (p-1)th layer and stops at 0th layer 
  
            d_ts[i] = 2*sum(1.0j*lam.conjugate().data@phi.evolve(derivativeM[i]).data for i in range(n)).real

            udag = core.reverse_mixing_operator(n,ts_guess[i])
            phi=phi.evolve(udag)

            lam = lam.evolve(udag)

            d_gamma_primes[i] = 2*(1j*lam.conjugate().data@(Qprime*phi.data)).real

            udag = core.reverse_phase_separator(n,Gprime,gamma_primes_guess[i])
            phi = phi.evolve(udag)

            lam = lam.evolve(udag)

            d_gammas[i] = 2*(1j*lam.conjugate().data@(Q*phi.data)).real

            udag = core.reverse_phase_separator(n,G,gammas_guess[i])
            phi = phi.evolve(udag)

            lam = lam.evolve(udag)
        
        return np.concatenate((d_gammas, d_gamma_primes, d_ts))

    if optimize:
        # Combine initial gamma and time parameters into one array
        initial_params = np.concatenate((gammas,gamma_primes, ts))
        
        # Default bounds if not provided
        if bounds is None:
            bounds = [(0, 2*np.pi)] * (3*p)  # Assuming ranges [0, 2π] for all parameters

        '''
        costs=[]
        def make_callback(costs):
            iteration = 0
            def callback(xk):
                nonlocal iteration
                iteration += 1
                obj_value = cost(xk)
                costs.append(obj_value)
                print(f"Iteration: {iteration},  Objective function value: {obj_value}")
            return callback

        callback = make_callback(costs)
        '''
        
        # Minimize the negative expectation value
        result = minimize(cost, initial_params, bounds=bounds, method=method,jac=gradients,options=options)#, callback=callback)
    
        #scipy_output = result
        num_iters = result.nit
        
        # Extract the best found parameters
        best_params = result.x
        best_gammas = best_params[:p]
        best_gamma_primes = best_params[p:2*p]
        best_ts = best_params[2*p:]
        
        # The maximum expectation value is the negative of the minimum found
        max_expectation_value = -result.fun
        approxn_ratio = max_expectation_value/max(Q)
        
        # Get the probablility of measuring the best value
        qc = map_params(best_gammas, best_gamma_primes,best_ts, qc, gamma_parameters,gamma_prime_parameters, t_parameters, p)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        prob_max = sum(probs[np.where(Q == max(Q))[0]]) #probability of measuring a maximum cut = sum of probability of each maximum cut
        
        return result, num_iters, approxn_ratio, prob_max, best_gammas, best_gamma_primes, best_ts
    
    else:
        
        # Calculate expectation with given parameters
        qc = map_params(gammas, gamma_primes, ts, qc, gamma_parameters, gamma_prime_parameters, t_parameters, p)
        statevector = backend.run(qc).result().get_statevector()
        probs = (np.abs(statevector)**2)
        if return_max_prob:
            return Q @ probs, sum(probs[np.where(Q == max(Q))[0]])
        else:
            return Q @ probs

# run the full VQA (with supplied initial parameters)
def run(backend, Q, G, Gprime, p, q, params_init):
    # set up
    #- cut sizes
    Q = core.qualities(G)
    
    #- prepare equal superposition and label parameters in PQC
    qc, gamma_parameters, gamma_prime_parameters, t_parameters = PQC_sim_setup(backend, G, Gprime, p)
    
    #- initialize parameters with pretrained params_init
    gammas = [0] * p # initialize gammas to 0
    gamma_primes = params_init[0]
    ts = params_init[1]
    
    # simulation
    scipy_output, num_iters, approxn_ratio, prob_max, best_gammas, best_gamma_primes, best_ts = sim_and_optimize(backend, Q, qc, G, Gprime, gamma_parameters, gamma_prime_parameters, t_parameters, gammas, gamma_primes, ts, p, optimize=True, options={"maxfun": 50000, "maxiter": q})

    return scipy_output, num_iters, approxn_ratio, prob_max, best_gammas, best_gamma_primes, best_ts
