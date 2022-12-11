"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Reports
"""

import numpy as np
import pandas as pd
import json

from Solvers import *

filename = "reports/reports.json"

with open(filename, "r") as f:
    data = json.load(f)

for function in constraints:

    print(function)

    def_func = constraints[function]['def_func']

    search_space = constraints[function]['search_space']

    design_space = constraints[function]['design_space']

    minimum = constraints[function]['minimum']

    defs = def_func()

    function_values = []

    best_fits = []

    column_names = ["Algorithm", "Minimal", "Maximum", "Mean", "Median"]

    best_fits_column_names = ["Algorithm", "Solution", "Iterations", "Function Evaluations", "Function Value"]

    for alg_name in algorithms:

        function_values_data = data[function][alg_name]["design_space"]["overall"]["function_values"]

        best_fit_data = data[function][alg_name]["design_space"]["best_fit"]["best_fit_data"]

        function_values.append([alg_name,
                                function_values_data["fmin"],
                                function_values_data["fmax"],
                                function_values_data["favg"],
                                function_values_data["fmedian"]])

        best_fits.append([alg_name,
                          ",".join([str(x) for x in best_fit_data["optimal_point"]]),
                          best_fit_data["iterations"],
                          best_fit_data["function_evaluations"],
                          best_fit_data["optimal_function_value"]])

    # function_data = np.array(function_values)

    df = pd.DataFrame(function_values, columns=column_names)

    content = df.to_latex(index=False)

    with open("latex/" + function + "_function_values.tex", "w") as f:
        f.write(content)

    # best_fit_data = np.array(best_fits)

    df = pd.DataFrame(best_fits, columns=best_fits_column_names)

    content = df.to_latex(index=False)

    with open("latex/" + function + "_best_fits.tex", "w") as f:
        f.write(content)
