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

    column_names = ["Alg.", "Min", "Max", "Mean", "Median"]

    best_fits_column_names = ["Alg.", "Sol.", "Iter.", "F. Eval", "F. Value"]

    solutions_rows = ['$x_{%s}$' % x for x in range(1, defs['dimension']+1)]

    solution_names = list()

    solution_names.append("Coord.")

    solutions = list()

    solutions.append(solutions_rows)

    convergence = list()

    convergence_names = ["Alg.", "Good", "Poor", "Diver.", "Total"]

    index = 1

    for alg_name in algorithms:

        good = data[function][alg_name]["design_space"]["good"]['runs']

        poor = data[function][alg_name]["design_space"]["poor"]['runs']

        divergence = data[function][alg_name]["design_space"]["divergence"]['runs']

        overall = data[function][alg_name]["design_space"]["overall"]['runs']

        convergence.append([alg_name, good, poor, divergence, overall])

        function_values_data = data[function][alg_name]["design_space"]["overall"]["function_values"]

        best_fit_data = data[function][alg_name]["design_space"]["best_fit"]["best_fit_data"]

        solutions.append(best_fit_data["optimal_point"])

        function_values.append([alg_name,
                                function_values_data["fmin"],
                                function_values_data["fmax"],
                                function_values_data["favg"],
                                function_values_data["fmedian"]])

        best_fits.append([alg_name,
                          '$S_{%s}$' % index,
                          best_fit_data["iterations"],
                          best_fit_data["function_evaluations"],
                          best_fit_data["optimal_function_value"]])

        solution_names.append('$S_{%s}$' % index)

        index += 1

    solutions = list(map(list, zip(*solutions)))

    df = pd.DataFrame(function_values,
                      columns=column_names)

    content = df.to_latex(index=False,
                          float_format="%.2f",
                          escape=False,
                          label="function_values:{:s}".format(function),
                          caption="Statistical Information about function values For {:s}".format(function_names[function]))

    with open("latex/" + function + "_function_values.tex", "w") as f:
        f.write(content)

    df = pd.DataFrame(best_fits,
                      columns=best_fits_column_names)

    content = df.to_latex(index=False,
                          float_format="%.2f",
                          escape=False,
                          label="solutions:{:s}".format(function),
                          caption="Best Fits For {:s}".format(function_names[function]))

    with open("latex/" + function + "_best_fits.tex", "w") as f:
        f.write(content)

    df = pd.DataFrame(solutions,
                      columns=solution_names)

    content = df.to_latex(index=False,
                          float_format="%.2f",
                          escape=False,
                          label="detailedsolutions:{:s}".format(function),
                          caption="Detailed Solutions For {:s}".format(function_names[function]))

    with open("latex/" + function + "_best_solutions.tex", "w") as f:
        f.write(content)

    df = pd.DataFrame(convergence,
                      columns=convergence_names)

    content = df.to_latex(index=False,
                          float_format="%.2f",
                          escape=False,
                          label="convergence:{:s}".format(function),
                          caption="Convergence Report For {:s}".format(function_names[function]))

    with open("latex/" + function + "_convergence.tex", "w") as f:
        f.write(content)

