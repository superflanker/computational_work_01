"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Isolines and convergence line of the best fit
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from Solvers import *


filename = "reports/reports.json"

with open(filename, "r") as f:
    data = json.load(f)

for function in constraints:

    def_func = constraints[function]['def_func']

    search_space = constraints[function]['isoline_space']

    design_space = constraints[function]['design_space']

    minimum = constraints[function]['minimum']

    defs = def_func()

    function_function = defs['function_function']

    if defs['dimension'] == 2:

        print(function)

        x = np.linspace(search_space[0], search_space[1], 500)

        y = np.linspace(search_space[0], search_space[1], 500)

        X, Y = np.meshgrid(x, y)

        Z = function_function(*[X, Y])

        levels = np.linspace(np.min(Z), np.max(Z), 100)

        plt.close('all')

        plt.contour(X, Y, Z, levels=levels, linewidths=[0.3])

        for alg_name in algorithms:

            best_fit = data[function][alg_name]["design_space"]["best_fit"]["best_fit_data"]

            #: x, y

            points = np.array(best_fit["points"])

            x = points[:, 0]
            y = points[:, 1]
            z = np.array(best_fit["function_values"])

            plt.plot(x, y, label=alg_name, linewidth=0.6)

        plt.title(function_names[function])

        plt.xlabel("X axis")

        plt.ylabel("Y axis")

        plt.legend()

        plt.grid()

        plt.savefig("images/" + function + ".jpg", dpi=600)