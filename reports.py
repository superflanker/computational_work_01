"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Reports
"""

import numpy as np
import json

from Solvers import *


def min_max_avg_std(data):
    """
    Computes min, max, avg and std from results
    :param data: the results data
    :return: min, max, avg, std, first_quantile and third_quantile values
    """
    fmin = 0
    fmax = 0
    favg = 0
    fmedian = 0
    fstd = 0

    if len(data) > 0:
        fmin = float(np.min(data))
        fmax = float(np.max(data))
        favg = float(np.mean(data))
        fmedian = float(np.median(data))
        fstd = float(np.std(data))

    return {"fmin": fmin,
            "fmax": fmax,
            "favg": favg,
            "fmedian": fmedian,
            "fstd": fstd}


def extract_data(data):
    """
    Data extraction
    :param data: the results data
    :param convergence: the convergence class (good, poor, divergence)
    :return: the extracted data
    """
    extracted_data = dict()
    extracted_data["good"] = {"data": [], "elapsed_time": []}
    extracted_data["poor"] = {"data": [], "elapsed_time": []}
    extracted_data["divergence"] = {"data": [], "elapsed_time": []}
    for i in range(0, len(data['runs']['data'])):
        node = data['runs']['data'][i]
        extracted_data[node['convergence']]["data"].append(node)
        extracted_data[node['convergence']]["elapsed_time"].append(data["elapsed_time"][i])

    return extracted_data


def extract_parameter(data, parameter):
    """
    Extract parameter from data
    :param data: the result data
    :param parameter: the parameter
    :return: the parameter values
    """
    parameter_values = list()
    for i in range(0, len(data)):
        parameter_values.append(float(data[i][parameter]))
    return parameter_values


def sortFunc(e):
    return e[1]


def best_fit(data, function_minimum):
    to_sort = list()
    for i in range(0, len(data)):
        sort_index = np.abs(data[i]['optimal_function_value'] - function_minimum)
        to_sort.append([i, sort_index])
    to_sort.sort(key=sortFunc)
    best_fit = to_sort[0][0]
    best_fit_distance = to_sort[0][1]
    best_fit_data = data[best_fit]

    return {"best_fit_index": int(best_fit),
            "best_fit_distance": float(best_fit_distance),
            "best_fit_data": best_fit_data}


results = dict()

for function in constraints:

    def_func = constraints[function]['def_func']

    search_space = constraints[function]['search_space']

    design_space = constraints[function]['design_space']

    minimum = constraints[function]['minimum']

    defs = def_func()

    function_minimum = defs['function_function'](*minimum)

    if function not in results:
        results[function] = dict()

    for alg_name in algorithms:

        results[function][alg_name] = dict()

        print("processing {:s} {:s}".format(function,
                                            alg_name))

        filenames = [["clean_results/" + alg_name + "_design_space_" + function + ".json", 'design_space'],
                     ["clean_results/" + alg_name + "_search_space_" + function + ".json", 'search_space']]

        for filename in filenames:
            search_space = filename[1]
            filename = filename[0]
            results[function][alg_name][search_space] = {"good": {},
                                                         "poor": {},
                                                         "divergence": {},
                                                         "overall":{},
                                                         "best_fit":{}}
            with open(filename, "r") as f:
                data = json.load(f)

            #: data extraction

            old_data = data.copy()

            data = extract_data(old_data)

            # convergence_class
            results[function][alg_name][search_space]['good']['runs'] = len(data['good']['data'])
            results[function][alg_name][search_space]['poor']['runs'] = len(data['poor']['data'])
            results[function][alg_name][search_space]['divergence']['runs'] = len(data['divergence']['data'])
            results[function][alg_name][search_space]['overall']['runs'] = len(old_data['runs']['data'])

            #: time elapsed
            results[function][alg_name][search_space]['good']['elapsed_time'] = min_max_avg_std(
                data['good']['elapsed_time'])
            results[function][alg_name][search_space]['poor']['elapsed_time'] = min_max_avg_std(
                data['poor']['elapsed_time'])
            results[function][alg_name][search_space]['divergence']['elapsed_time'] = min_max_avg_std(
                data['divergence']['elapsed_time'])
            results[function][alg_name][search_space]['overall']['elapsed_time'] = min_max_avg_std(
                old_data['elapsed_time'])

            # iterations
            results[function][alg_name][search_space]['good']['iterations'] = min_max_avg_std(extract_parameter(
                data['good']['data'], 'iterations'))
            results[function][alg_name][search_space]['poor']['iterations'] = min_max_avg_std(extract_parameter(
                data['poor']['data'], 'iterations'))
            results[function][alg_name][search_space]['divergence']['iterations'] = min_max_avg_std(extract_parameter(
                data['divergence']['data'], 'iterations'))
            results[function][alg_name][search_space]['overall']['iterations'] = min_max_avg_std(extract_parameter(
                old_data['runs']['data'], 'iterations'))

            # function evaluations
            results[function][alg_name][search_space]['good']['function_evaluations'] = min_max_avg_std(
                extract_parameter(
                    data['good']['data'], 'function_evaluations'))
            results[function][alg_name][search_space]['poor']['function_evaluations'] = min_max_avg_std(
                extract_parameter(
                    data['poor']['data'], 'function_evaluations'))
            results[function][alg_name][search_space]['divergence']['function_evaluations'] = min_max_avg_std(
                extract_parameter(
                    data['divergence']['data'], 'function_evaluations'))
            results[function][alg_name][search_space]['overall']['function_evaluations'] = min_max_avg_std(
                extract_parameter(old_data['runs']['data'], 'function_evaluations'))

            # gradient evaluations
            results[function][alg_name][search_space]['good']['gradient_evaluations'] = min_max_avg_std(
                extract_parameter(
                    data['good']['data'], 'gradient_evaluations'))
            results[function][alg_name][search_space]['poor']['gradient_evaluations'] = min_max_avg_std(
                extract_parameter(
                    data['poor']['data'], 'gradient_evaluations'))
            results[function][alg_name][search_space]['divergence']['gradient_evaluations'] = min_max_avg_std(
                extract_parameter(
                    data['divergence']['data'], 'gradient_evaluations'))
            results[function][alg_name][search_space]['overall']['gradient_evaluations'] = min_max_avg_std(
                extract_parameter(
                    old_data['runs']['data'], 'gradient_evaluations'))

            # function values
            results[function][alg_name][search_space]['good']['function_values'] = min_max_avg_std(extract_parameter(
                data['good']['data'], 'optimal_function_value'))
            results[function][alg_name][search_space]['poor']['function_values'] = min_max_avg_std(extract_parameter(
                data['poor']['data'], 'optimal_function_value'))
            results[function][alg_name][search_space]['divergence']['function_values'] = min_max_avg_std(
                extract_parameter(
                    data['divergence']['data'], 'optimal_function_value'))
            results[function][alg_name][search_space]['overall']['function_values'] = min_max_avg_std(
                extract_parameter(
                    old_data['runs']['data'], 'optimal_function_value'))

            #: best fit
            results[function][alg_name][search_space]['best_fit'] = best_fit(old_data['runs']['data'],
                                                                             function_minimum)


with open("reports/reports.json", "w") as f:
    json.dump(results, f, indent=4)

