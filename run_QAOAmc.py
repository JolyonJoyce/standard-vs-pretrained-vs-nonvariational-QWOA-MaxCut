import argparse
import standard_QAOAmc as stdQAOAmc
import pretrained_QAOAmc as ptQAOAmc
import nonvariational_QWOAmc as nvQWOAmc 
import numpy as np
import networkx as nx
from time import time
from qiskit_aer import AerSimulator
from QAOAmc_core_components import generate_graph, qualities

# standard QAOAmc, small-DLA pretrained QAOAmc, and non-variational QWOAmc

#use flags to set the number of qubits, depth of the circuit and number of iterations
#recomended to set max iters~10 for a runtime of around 1 hour. Note to run
#a depth other than 256 and 16 the pretraining must first be completed and an approporiate file
# with initial parameters created

#the run time scales as roughly 3mins~1 iteration on a personal deviceTo achieve
#better results this hsould be set to around 500 (will usually only take ~250)

# to generate the 400 graphs used in Matthaus' thesis:
# - set edge_prob (-e) to 0.3
# - graph_type (-G) to "ER" or "pseudo-3-reg"
# - graph_seed (-s) to 0 through 199

def main_args():
    parser = argparse.ArgumentParser(
        description="Simulate QAOA-MaxCut circuit for MaxCut on a random graph.",
        epilog="",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        default="std",
        type=str,
        dest="alg",
        nargs="?",
        help="""Choose algorithm to run:
        - std : non-variational QWOA-MaxCut
        - nv : standard QAOA-MaxCut
        - pt : small-DLA pre-trained QAOA-MaxCut
        - all : run all in succession
        default = std""",
    )
    parser.add_argument(
        "-n",
        "--nqubits",
        default=4,
        type=int,
        dest="n",
        nargs="?",
        help="number of vertices, default = 4", # we will use n = 16
    )
    parser.add_argument(
        "-p",
        "--PQCdepth",
        default=1,
        type=int,
        dest="p",
        nargs="?",
        help="number of ansatz layers, default = 1", #we will use p = 256
    )
    parser.add_argument(
        "-q",
        "--tuning_iterations",
        default=1,
        type=int,
        dest="q",
        nargs="?",
        help="number of iterations for classical tuning, default = 1", # we will use q = 500
    )
    parser.add_argument(
        "-B",
        "--blocking-qubits",
        default=0,
        type=float,
        dest="blocking_qubits",
        nargs="?",
        help="size of qubit blocks for MPI parallelization, 0 sets blocking_enabled=False, default = 0",
    )
    parser.add_argument(
        "-T",
        "--threshold",
        default="14",
        dest="threshold",
        type=str,
        nargs="?",
        help="number of qubits at which to use parallelize matrix multiplication with OpenMp",
    )
    parser.add_argument(
        "-e",
        "--edge-prob", 
        default="0.3",
        dest="edge_prob",
        type=float,
        nargs="?",
        help="""the probability of edge generation for ER graphs (-G ER), default = 0.3.
        N.B. only affects graph generation when --graph-type = ER""",
    )
    parser.add_argument(
        "-s",
        "--graph_seed",
        default=None,
        dest="graph_seed",
        type=int,
        nargs="?",
        help="seed for random target graph generation, default = None",
    )
    parser.add_argument(
        "-w",
        "--weighted",
        default=False,
        dest="weighted",
        type=bool,
        nargs="?",
        help="whether target graph is weighted (True) or unweighted (False), default = False",
    )    
    parser.add_argument(
        "-G",
        "--graph-type",
        default="ER",
        dest="graph_type",
        type=str,
        nargs="?",
        help="""type of target graph:
        - ER : a random graph whose edges are randomly generated based on the supplied edge probability (-e)
        - 3-reg : a randomly weighted 3-regular graph
        - pseudo-3-reg : a dubiously generated randomly weighted 3-regular graph (very small probability of non-3-regular graph)
        - path : the path graph with all weights = 1
        - cycle : the cycle graph with all weights = 1
        - complete : the complete graph with all weights = 1
        """,
    )
    parser.add_argument(
        "-A",
        "--aux-graph",
        default="path",
        dest="aux_graph",
        type=str,
        nargs="?",
        help="""type of auxillary graph for ptQAOAmc:
        - path OR line : path graph
        - cyc OR cycle OR cyclic : cycle graph
        - comp OR compl OR complete : complete graph
        default = path""",
    )
    parser.add_argument(
        "-W",
        "--write-params",
        default="",
        dest="optimal_params_file",
        type=str,
        nargs="?",
        help="supply filename (including .npz extension) into which to write optimal parameters for stdQAOAmc; if empty, optimal parameters will not be written to a file; default = (empty)",
    )
    parser.add_argument(
        "-I",
        "--import-params",
        default="",
        dest="initial_params_file",
        type=str,
        nargs="?",
        help="supply pretrained initial parameters for ptQAOAmc from a file by giving its relative filename (including .npz extension); if empty, random initial parcustom (usually pretrained) ameters will be used; default = (empty)",
    )
    parser.add_argument(
        "-D",
        "--device",
        default="CPU",
        dest="device",
        type=str,
        nargs="?",
        help="device to run on, default = CPU",
    )
    parser.add_argument(
        "-l",
        "--logs",
        default=True,
        dest="logs",
        type=bool,
        nargs="?",
        help="whether to print detailed logs (default = True)"
    )
    parser.add_argument(
        "-b",
        "--beta-init",
        default=False,
        dest="beta-init",
        type=float,
        nargs="?",
        help=""
    )
    parser.add_argument(
        "-g",
        "--gamma-init",
        default=False,
        dest="gamma-init",
        type=float,
        nargs="?",
        help=""
    )
    parser.add_argument(
        "-t",
        "--t-init",
        default=False,
        dest="t-init",
        type=float,
        nargs="?",
        help=""
    )

    return parser.parse_args()

if __name__ == "__main__":
    # import arguments
    args = main_args()

    alg = args.alg
    if alg not in ["std", "pt", "nv", "all"]:
        raise ValueError(f"Unknown algorithm type: {alg}")
    
    n = args.n
    p = args.p
    q = args.q
    
    aux_graph = args.aux_graph
    
    edge_prob = args.edge_prob
    graph_seed = args.graph_seed
    weighted = args.weighted
    graph_type = args.graph_type
    
    optimal_params_file = args.optimal_params_file
    initial_params_file = args.initial_params_file
    
    blocking_qubits = args.blocking_qubits
    threshold = args.threshold
    device = args.device
    
    logs = args.logs

    print("Running...")
    
    if blocking_qubits == 0:
        blocking_enabled = False
    else:
        blocking_enabled = True

    backend_options={
        "device": device,
        "method": "statevector",
        "blocking_enable": blocking_enabled,
        "blocking_qubits": blocking_qubits,
        "cuStateVec_enable": True
    }

    backend = AerSimulator(**backend_options)
    
    #load the parameters from the file supplied by -I
    if initial_params_file != "":
        loaded = np.load(initial_params_file)
        params_init = np.array([loaded['gammas'], loaded['ts']])
        if logs:
            print("Successfully loaded initial parameters for ptQAOAmc from ", initial_params_file)
            print()
    
    if logs:
        num_edges_complete = round(n*(n-1)/2)
    
    if alg == "pt":
        Gprime = generate_graph(n, aux_graph, weighted=False)
        if logs:
            print(f"Successfully generated auxillary graph G\' with data:")
            print(f" ‣ unweighted {aux_graph} graph on n={n} vertices")
            
            num_edges_prime = round(np.count_nonzero(Gprime)/2)
            
            print(" ‣ number of edges =", num_edges_prime)
            print(f" ‣ density = [number of edges] / [n(n-1)/2] = {num_edges_prime}/{num_edges_complete}", "=", num_edges_prime/num_edges_complete)
            #print(" ‣ adjacency matrix:")
            #print(Gprime)
            print()

    G = generate_graph(n, graph_type, weighted, edge_prob, graph_seed)
    Q = qualities(G)
    if logs:
        print()
        print(f"Successfully generated target graph G with data:")
    
        if weighted == False: w = "unweighted"
        else: w = "randomly weighted"
        
        if graph_type == "ER": print(f" ‣ {w} Erdős-Rényi random graph on n={n} vertices with edge generation probability of", edge_prob)
        else: print(f" ‣ {w} {graph_type} graph on n={n} vertices")

        num_edges = round(np.count_nonzero(G)/2)
        print(" ‣ number of edges =", num_edges)
        print(f" ‣ density = [number of edges] / [n(n-1)/2] = {num_edges}/{num_edges_complete}", "=", num_edges/num_edges_complete)
        print(" ‣ (weighted) adjacency matrix:")
        print(G)
    
        #analyze cuts (possible solutions and their qualities, i.e. cut sizes)
        print(" ‣ distribution of cut sizes (objective function values):")
        #print(f"   ‣ number of cuts = 2^n = {2**n}")
        print("   ‣ max  = ", max(Q))
        print("   ‣ mean = ", np.mean(Q))
        print("   ‣ med  = ", np.median(Q))
        print("   ‣ std  = ", np.std(Q))
        print()
    
    #run standard QAOA
    if alg == 'std' or alg == 'all':
        
        if logs:
            print(f"Running stdQAOAmc(G,{p},{q})...")
            print()
            print("!! -1 * (expectation value of cut size) at each iteration:")
            start_time = time()
        
        scipy_output, num_iters, expectation_ratio, prob_max, best_gammas, best_ts = stdQAOAmc.run(backend, Q, G, p, q)
        
        if logs:
            end_time = time()
            #print("Output from scipy.optimize.minimize:")
            #print(scipy_output)
            print()
            print(f"Completed stdQAOAmc(G,{p},{q}) with optimization terminating after", num_iters, "iterations, yielding:")
            print(" ‣ expectation ratio =", expectation_ratio, "( = [minimum found expectation value of cut size of amplified state] / [maximum cut size] )")
            print(" ‣ prob of measuring a maximum cut =", prob_max)
            print()
            print(f"Time elapsed for hybrid algorithm:", end_time - start_time)
        
        if optimal_params_file != "":
            np.savez(optimal_params_file, gammas=best_gammas, ts=best_ts)
            if logs:
                print(f"Successfully wrote optimal parameters to file {optimal_params_file}")
            # write out best_gammas and best_ts to .npz file to be loaded as follows:
            #   loaded = np.load(initial_params_file)
            #   params_init = np.array([np.array(loaded[key]) for key in loaded])
        
    #run pretrained QAOA
    elif alg == 'pt' or alg == 'all':
        if logs:
            print(f"Running ptQAOAmc(G,{p},{q})...")
            print()
            print("!! -1 * (expectation value of cut size) at each iteration:")
            start_time = time()
        
        scipy_output, num_iters, expectation_ratio, prob_max, best_gammas, best_gamma_primes, best_ts = ptQAOAmc.run(backend, Q, G, Gprime, p, q, params_init)
        end_time = time()
        
        if logs:
            #print("Output from scipy.optimize.minimize:")
            #print(scipy_output)
            print()
            print(f"Completed ptQAOAmc(G,{p},{q}) with optimization terminating after", num_iters, "iterations, yielding:")
            print(" ‣ expectation ratio =", expectation_ratio, "( = [minimum found expectation value of cut size of amplified state] / [maximum cut size] )")
            print(" ‣ prob of measuring a maximum cut =", prob_max)
            print()
            print(f"Time elapsed for hybrid algorithm:", end_time - start_time)

    #run non-variational QWOA
    elif alg == 'nv' or alg == 'all':
        if logs:
            print(f"Running nvQWOAmc(G,{p},{q})...")
            print()
            print("!! -1 * (expectation value of cut size) at each iteration:")
            start_time = time()
        
        scipy_output, num_iters, expectation_ratio, prob_max, best_beta, best_gamma, best_t = nvQWOAmc.run(backend, Q, G, p, q)
        
        if logs:    
            end_time = time()
            #print("Output from scipy.optimize.minimize:")
            #print(scipy_output)
            print()
            print(f"Completed nvQAOAmc(G,{p},{q}) with optimization terminating after", num_iters, "iterations, yielding:")
            print(" ‣ expectation ratio =", expectation_ratio, "( = [minimum found expectation value of cut size of amplified state] / [maximum cut size] )")
            print(" ‣ prob of measuring a maximum cut =", prob_max)
            print(f" ‣ best found parameters: (β,γ,t) = ({best_beta:.6}, {best_gamma:.6}, {best_t:.6})")
            print()
            print(f"Time elapsed for hybrid algorithm:", end_time - start_time)
    
    if logs:    
        print()
