"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Common functions used by the whole system
"""

from sympy import latex, var, diff, simplify, lambdify
import numpy as np
import scipy.optimize as optimizer

"""
Sympy definitions for gradient and hessian functions
"""


def get_gradient_vector(diff_vars, function):
    """
    Analytical Gradient Vector
    :param diff_vars: differentiation vars
    :param function: differentiable function
    :return: list the gradient
    """
    gradient = list()
    for i in range(len(diff_vars)):
        gradient.append(diff(function, diff_vars[i]))
    return gradient


def get_hessian_matrix(diff_vars, function):
    """
    Analytical Hessian Matrix
    :param diff_vars: differentiation vars
    :param function: differentiable function
    :return: list the hessian matrix
    """
    hessian = list()
    for i in range(len(diff_vars)):
        line = list()
        for j in range(len(diff_vars)):
            line.append(diff(diff(function, diff_vars[i]), diff_vars[j]))
        hessian.append(line)
    return hessian


def get_lambdified_function(diff_vars, function):
    """
    Transforms the function into python function
    :param diff_vars: differentiation vars
    :param function: differentiable function
    :return: fun the function function
    """
    return lambdify(diff_vars, function, "numpy")


def get_lambdified_gradient_vector(diff_vars, function):
    """
    Transforms the analytical gradient into python function
    :param diff_vars: differentiation vars
    :param function: differentiable function
    :return: fun the gradient function
    """
    return lambdify(diff_vars, get_gradient_vector(diff_vars, function), "numpy")


def get_lambdified_hessian_matrix(diff_vars, function):
    """
    Transforms the analytical hessian matrix into python function
    :param diff_vars: differentiation vars
    :param function: differentiable function
    :return: list the hessian matrix
    """
    return lambdify(diff_vars, get_hessian_matrix(diff_vars, function), "numpy")


def generate_random_sequence(seed, lmin, lmax, n):
    """
    Random sequence generator (for initial guess generation)
    :param seed: random seed
    :param lmin: lower bound
    :param lmax: high bund
    :param n: sequence lenght
    :return: array the random sequence
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(lmin, lmax, n)


"""
Objective function, gradient and hessian analytical and estimation procedures
"""

def evaluate_objective_function(f, xvec, stats):
    """
    Function Evaluation
    :param f: the evaluated function
    :param xvec: x point for evaluation
    :return: scalar/vector/matrix the evaluated value/array
    """
    # first things first
    if 'objective_function_calls' not in stats:
        stats['objective_function_calls'] = 0
    stats['objective_function_calls'] += 1
    return np.array(f(*xvec))


def compute_gradient(gradF, x, stats):
    """
    Compute gradient using the analytical way
    :param gradF: the gradient function, given gladly to us using
    sympy diff and lambdify functions
    :param x: the evaluation point
    :param stats: function calls stats
    :return: the computed gradient
    """
    # first things first
    if 'gradient_calls' not in stats:
        stats['gradient_calls'] = 0
    stats['gradient_calls'] += 1
    return np.array(gradF(*x))


def compute_hessian(hessF, x, stats):
    """
    Compute Hessian Matrix using the analytical way
    :param hessF: the Hessian Function Matrix
    :param x: the point to be evaluated
    :param stats: function calls stats
    :return: the numerical hessian matrix
    """
    # first things first
    if 'hessian_function_calls' not in stats:
        stats['hessian_function_calls'] = 0
    stats['hessian_function_calls'] += 1
    return np.array(hessF(*x))

