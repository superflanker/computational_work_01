"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

app runner
"""

import jdata as jd
import json
import time
from Solvers import *

N = 100


def solve(alg_name, alg_func, function, search_space, defs):
    print("Solving for {:s} - {:s}".format(alg_name, function))

    results = list()
    times = list()
    for seed in range(1, N + 1):
        start = time.time()

        x0 = generate_random_sequence(seed,
                                      search_space[0],
                                      search_space[1],
                                      defs['dimension'])

        results.append(alg_func(defs['function_function'],
                                defs['gradient_function'],
                                defs['hessian_function'],
                                x0))
        stop = time.time()

        times.append(stop - start)

    results = {"runs": results,
               "elapsed_time": times}

    filename = "results/" + alg_name + "_" + function + ".json"

    #: jdata saves very complex data structures directly in json
    jd.save(results, filename)

    #: some formatting that jdata doesn't handle by itself
    with open(filename, "r") as f:
        json_data = json.load(f)

    with open(filename, "w") as f:
        json.dump(json_data, f, indent=4)


for function in constraints:

    def_func = constraints[function]['def_func']

    search_space = constraints[function]['search_space']

    design_space = constraints[function]['design_space']

    defs = def_func()

    for alg_name in algorithms:
        alg_func = algorithms[alg_name]
        solve(alg_name + "_search_space", alg_func, function, search_space, defs)
        solve(alg_name + "_design_space", alg_func, function, design_space, defs)
