#!/usr/bin/env python3

"""
From the original author, Michal Stechly:
    Finding a solution to the travelling salesman problem.
    This code is based on the following project made by BOHR.TECHNOLOGY:
    https://github.com/BOHRTECHNOLOGY/quantum_tsp
    Which was in turn based on the articles by Stuart Hadfield:
    https://arxiv.org/pdf/1709.03489.pdf
    https://arxiv.org/pdf/1805.03265.pdf

The original implementation of this file
(found at  https://github.com/mstechly/grove/tree/master/grove/pyqaoa) has been
updated for use in our CS 269Q final project.
"""

import pyquil.api as api
from grove.pyqaoa.qaoa import QAOA
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
from pyquil.gates import X
import scipy.optimize
import numpy as np

import sys
from classical import *
from IPython.core.debugger import set_trace

def solve_tsp(weights, connection=None, steps=1, ftol=1.0e-4, xtol=1.0e-4, samples=1000):
    """
    Method for solving travelling salesman problem.
    :param weights: Weight matrix. weights[i, j] = cost to traverse edge from city i to j. If
        weights[i, j] = 0, then no edge exists between cities i and j.
    :param connection: (Optional) connection to the QVM. Default is None.
    :param steps: (Optional. Default=1) Trotterization order for the QAOA algorithm.
    :param ftol: (Optional. Default=1.0e-4) ftol parameter for the Nelder-Mead optimizer
    :param xtol: (Optional. Default=1.0e-4) xtol parameter for the Nelder-Mead optimizer
    :param samples: (Optional. Default=1000) number of bit string samples to use.
    """
    if connection is None:
        connection = api.QVMConnection()
    number_of_cities = weights.shape[0]
    list_of_qubits = list(range(number_of_cities**2))
    number_of_qubits = len(list_of_qubits)
    cost_operators = create_cost_hamiltonian(weights)
    driver_operators = create_mixer_operators(number_of_cities)
    initial_state_program = create_initial_state_program(number_of_cities)

    minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': ftol, 'xtol': xtol,
                                        'disp': False}}

    vqe_option = {'disp': print_fun, 'return_all': True,
                  'samples': None}

    qaoa_inst = QAOA(connection, list_of_qubits, steps=steps, cost_ham=cost_operators,
                     ref_ham=driver_operators, driver_ref=initial_state_program, store_basis=True,
                     minimizer=scipy.optimize.minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options=vqe_option)

    betas, gammas = qaoa_inst.get_angles()
    most_frequent_string, _ = qaoa_inst.get_string(betas, gammas, samples=samples)
    solution = binary_state_to_points_order(most_frequent_string)
    return solution

def create_cost_hamiltonian(weights):
    """
    Translating the distances between cities into the cost hamiltonian.
    """
    cost_operators = []
    number_of_cities = weights.shape[0]
    for t in range(number_of_cities - 1):
        for city_1 in range(number_of_cities):
            for city_2 in range(number_of_cities):

                # If these aren't the same city and they have an edge connecting them
                distance = weights[city_1, city_2]
                if city_1 != city_2 and distance != 0.0:
                    qubit_1 = t * number_of_cities + city_1
                    qubit_2 = (t + 1) * number_of_cities + city_2
                    cost_operators.append(PauliTerm("Z", qubit_1, distance) * PauliTerm("Z", qubit_2))
    cost_hamiltonian = [PauliSum(cost_operators)]
    return cost_hamiltonian

def create_mixer_operators(n):
    """
    Creates mixer operators for the QAOA.
    It's based on equations 54 - 58 from https://arxiv.org/pdf/1709.03489.pdf
    Indexing here comes directly from section 4.1.2 from paper 1709.03489, equations 54 - 58.
    """
    mixer_operators = []
    for t in range(n - 1):
        for city_1 in range(n):
            for city_2 in range(n):
                i = t
                u = city_1
                v = city_2
                first_part = 1
                first_part *= s_plus(n, u, i)
                first_part *= s_plus(n, v, i+1)
                first_part *= s_minus(n, u, i+1)
                first_part *= s_minus(n, v, i)

                second_part = 1
                second_part *= s_minus(n, u, i)
                second_part *= s_minus(n, v, i+1)
                second_part *= s_plus(n, u, i+1)
                second_part *= s_plus(n, v, i)
                mixer_operators.append(first_part + second_part)
    return mixer_operators

def create_initial_state_program(number_of_cities):
    """
    This creates a state, where at t=i we visit i-th city.
    """
    initial_state = Program()
    for i in range(number_of_cities):
        initial_state.inst(X(i * number_of_cities + i))

    return initial_state

def s_plus(number_of_cities, city, time):
    qubit = time * number_of_cities + city
    return PauliTerm("X", qubit) + PauliTerm("Y", qubit, 1j)

def s_minus(number_of_cities, city, time):
    qubit = time * number_of_cities + city
    return PauliTerm("X", qubit) - PauliTerm("Y", qubit, 1j)

def binary_state_to_points_order(binary_state):
    """
    Transforms the the order of points from the binary representation: [1,0,0,0,1,0,0,0,1],
    to the binary one: [0, 1, 2]
    """
    points_order = []
    number_of_points = int(np.sqrt(len(binary_state)))
    for p in range(number_of_points):
        for j in range(number_of_points):
            if binary_state[(number_of_points) * p + j] == 1:
                points_order.append(j)
    return points_order

def print_fun(x):
    print(x)

def path_cost(path, weights):
    """
    Calculates the cost of a path given the weight matrix.
    """
    path_len = len(path)
    cost = 0
    for i, city_ind in enumerate(path):
        if i == path_len - 1: break
        cost += weights[city_ind, path[i+1]]
    return cost

if __name__ == "__main__":
    """
    sys.argv ARGS:
        NUM_CITIES (int): number of cities in graph
        PERCENT_CONNECTED (float): percent of connections in graph
        STEPS (int): Trotterization order for QAOA

    # Example command line call:
    python tsp_qaoa_updated.py 3 1.0 1
    """

    # Very slow for NUM_CITIES >= 4. Need to change STEPS = 2 --> 1. why so slow for larger graphs?
    name, NUM_CITIES, PERCENT_CONNECTED, STEPS = sys.argv
    SAMPLES = 500 # Number of samples to use when finding most frequent path.
    weights, _ = rand_graph(int(NUM_CITIES), percent_connected = float(PERCENT_CONNECTED))
    solution   = solve_tsp(weights, steps=int(STEPS), xtol=10e-2, ftol=10e-2, samples=SAMPLES)

    print("Weight matrix")
    print(weights)
    print("-------------")
    print("The most frequent QAOA solution:")
    print("({}, {})".format(path_cost(solution, weights), solution))
    print("-------------")

    # Double check using the brute force classical solution:
    CHECK_CLASSICAL = True
    if CHECK_CLASSICAL:
        print("Classical solution:")
        print(classical(weights, _, loop=False))

    print('Number of cities = {}'.format(NUM_CITIES))
    print('Percent connected = {}'.format(PERCENT_CONNECTED))
    print('Trotterization order = {}'.format(STEPS))
