"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Data Cleanup
"""

import json
import numpy as np
from Solvers import *


def args_are_in_range(args, search_space):
    """
    Solution verifier
    :param args: the coordinates
    :param search_space: the search space
    :return: the in_range flag
    """
    in_search_space = True
    for arg in args:
        if arg < search_space[0] or arg > search_space[1]:
            in_search_space = False
    return in_search_space


def is_in_trust_region(x, minimum):
    """
    checks if the solution is in the strusted region for solution
    :param x: the solution
    :param minimum: the analytical minimum
    :return: bool the trust region flag
    """
    result = np.array(x) - np.array(minimum)
    norm = np.linalg.norm(result)
    return norm < 1e-2


for function in constraints:

    def_func = constraints[function]['def_func']

    search_space = constraints[function]['search_space']

    design_space = constraints[function]['design_space']

    minimum = constraints[function]['minimum']

    defs = def_func()

    for alg_name in algorithms:

        #: defining filename

        print("processing {:s} {:s}".format(function, alg_name))

        filenames = [["results/" + alg_name + "_design_space_" + function + ".json", design_space],
                     ["results/" + alg_name + "_search_space_" + function + ".json", design_space]]

        for filename in filenames:

            with open(filename[0], "r") as f:
                results = json.load(f)

            alg_data = results["runs"]

            elapsed_times = results["elapsed_time"]

            #: cleaning up file

            clean_data = dict()

            clean_data["hits"] = 0

            clean_data["miss"] = 0

            clean_data["out_of_range"] = 0

            clean_data["data"] = list()

            for i in  range(0, len(elapsed_times)):

                node = alg_data[i]

                new_node = dict()

                x = node["x"]["_ArrayData_"]

                new_node["iterations"] = node["nit"]

                new_node["function_evaluations"] = node["nfev"]

                new_node["gradient_evaluations"] = node["njev"]

                new_node["optimal_point"] = x

                new_node["optimal_function_value"] = node["fun"]

                new_node["points"] = list()

                new_node["function_values"] = list()

                for vector in node["allvecs"]:
                    x_data = vector["_ArrayData_"]
                    function_value = defs["function_function"](*x_data)
                    new_node["points"].append(x_data)
                    new_node["function_values"].append(function_value)

                if is_in_trust_region(x, minimum):

                    clean_data["hits"] += 1

                    new_node["convergence"] = 'good'

                else:

                    clean_data["miss"] += 1

                    new_node["convergence"] = 'poor'

                if not args_are_in_range(x, search_space):

                    clean_data["out_of_range"] += 1

                    new_node["convergence"] = 'divergence'

                clean_data["data"].append(new_node)

            clean_data = {"runs": clean_data,
                          "elapsed_time": elapsed_times}

            fname = filename[0].replace("results", "clean_results")

            with open(fname, "w") as f:
                json.dump(clean_data, f, indent=4)
