"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

ackley function and its derivatives
"""

from sympy import var, exp, cos, pi, euler, sqrt
from .common import *


def get_ackley_defs():

    x, y = var("x y")

    X = [x, y]

    function = - 20 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) \
               - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) \
               + exp(1) + 20

    function_function = get_lambdified_function(X, function)

    gradient = get_gradient_vector(X, function)

    gradient_function = get_lambdified_gradient_vector(X, function)

    hessian = get_hessian_matrix(X, function)

    hessian_function = get_lambdified_hessian_matrix(X, function)

    return {"function": function,
            "gradient": gradient,
            "hessian": hessian,
            "function_function": function_function,
            "gradient_function": gradient_function,
            "hessian_function": hessian_function,
            "diff_vars": X,
            "dimension": len(X)}
