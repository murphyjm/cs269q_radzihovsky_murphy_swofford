#!/usr/bin/env python3

import pyquil.api as api
from classical import rand_graph, classical, bitstring_to_path, calc_cost
from pyquil.paulis import sI, sZ, sX, exponentiate_commuting_pauli_sum
from scipy.optimize import minimize
from pyquil.api import WavefunctionSimulator
from pyquil.gates import H
from pyquil import Program

from collections import Counter

from tsp_qaoa_updated import binary_state_to_points_order

from IPython.core.debugger import set_trace
# import subprocess
# subprocess.Popen("/src/qvm/qvm -S > qvm.log 2>&1", shell=True)

# returns the bit index for an alpha and j
def bit(alpha, j):
    return j * num_cities + alpha

def D(alpha, j):
    b = bit(alpha, j)
    return .5 * (sI(b) - sZ(b))

"""
weights and connections are numpy matrixes that define a weighted and unweighted graph
penalty: how much to penalize longer routes (penalty=0 pays no attention to weight matrix)
"""
def build_cost(penalty, num_cities, weights, connections):
    ret = 0
    # constraint (a)
    for i in range(num_cities):
        cur = sI()
        for j in range(num_cities):
            cur -= D(i, j)
        ret += cur**2

    # constraint (b)
    for i in range(num_cities):
        cur = sI()
        for j in range(num_cities):
            cur -= D(j, i)
        ret += cur**2

    # constraint (c)
    for i in range(num_cities-1):
        cur = sI()
        for j in range(num_cities):
            for k in range(num_cities):
                if connections[j, k]:
                    cur -= D(j, i) * D(k, i+1)
        ret += cur

    # constraint (d) (the weighting)
    for i in range(num_cities-1):
        cur = sI()
        for j in range(num_cities):
            for k in range(num_cities):
                if connections[j, k]:
                    cur -= D(j, i) * D(k, i+1) * weights[j, k]
        ret += cur * penalty
    return ret

def qaoa_ansatz(betas, gammas, h_cost, h_driver):
    pq = Program()
    pq += [exponentiate_commuting_pauli_sum(h_cost)(g) + exponentiate_commuting_pauli_sum(h_driver)(b) for g,b in zip(gammas,betas)]
    return pq

def qaoa_cost(params, h_cost, h_driver, init_state_prog):
    half = int(len(params)/2)
    betas, gammas = params[:half], params[half:]
    program = init_state_prog + qaoa_ansatz(betas, gammas, h_cost, h_driver)
    return WavefunctionSimulator().expectation(prep_prog=program, pauli_terms=h_cost)


if __name__ == '__main__':

    num_cities = 3

    best_path = None
    best_cost = float("inf")
    weights, connections = None, None

    while not best_path:
        weights, connections = rand_graph(num_cities)
        best_cost, best_path = classical(weights, connections, loop=False)

    num_bits = num_cities**2
    num_params = 4

    # Optimize parameters
    init_state_prog = sum([H(i) for i in range(num_bits)], Program())
    penalty = .01
    h_cost = build_cost(penalty, num_cities, weights, connections)
    h_driver = -1. * sum(sX(i) for i in range(num_bits))
    result = minimize(qaoa_cost, x0=[.5]*num_params, method='Nelder-Mead')
    betas, gammas = result.x[0:int(num_params/2)], result.x[int(num_params/2):]

    qvm = api.QVMConnection()

    # Sample the circuit for the most frequent path i.e. least costly
    SAMPLES = 1000 # Number of samples
    bitstring_samples = qvm.run_and_measure(qaoa_ansatz(betas, gammas, h_cost, h_driver), list(range(num_bits)), trials=SAMPLES)

    best_quantum_cost = float("inf")
    best_quantum_path = None
    best_bitstring = None
    for bitstring in bitstring_samples:
        path = bitstring_to_path(bitstring)
        if not path:
            continue
        cost = calc_cost(path, weights, connections)
        if cost < best_quantum_cost:
            best_quantum_cost = cost
            best_quantum_path = path
            best_bitstring = bitstring
    print("Best quantum path")
    print(best_quantum_cost)
    print(best_quantum_path)
    print(best_bitstring)

    bitstring_tuples = list(map(tuple, bitstring_samples))
    freq = Counter(bitstring_tuples)
    print(freq)
    most_frequent_bit_string = max(freq, key=lambda x: freq[x])
    solution = binary_state_to_points_order(most_frequent_bit_string)

    print("number of occurances of our bitstring: ", freq[tuple(best_bitstring)])

    print("Weights")
    print(weights)
    print("-------------")
    print("Most frequent QAOA path (and bit string):")
    print(solution)
    print(most_frequent_bit_string)
    print("-------------")
    print("Classical solution:")
    print("({}, {})".format(best_cost, best_path))
