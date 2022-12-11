"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Rosenbrock 4D function and its derivatives
"""

from sympy import var
from .common import *


def get_rosenbrock4D_defs():

    x1, x2, x3, x4 = var("x1 x2 x3 x4")

    X = [x1, x2, x3, x4]

    function = 100 * ((x2 - x1 ** 2) ** 2 +
                      (x3 - x2 ** 2) ** 2 +
                      (x4 - x3 ** 2) ** 2) +\
               ((x1 - 1) ** 2 +
                (x2 - 1) ** 2 +
                (x3 - 1) ** 2)

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
