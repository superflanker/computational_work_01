"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Rastrigin 2D function and its derivatives
"""

from sympy import var, cos, sin, pi
from .common import *


def __get_rastrigin_function():
    A = 10

    (x1,
     x2,
     x3,
     x4,
     x5,
     x6,
     x7,
     x8,
     x9,
     x10,
     x11,
     x12,
     x13,
     x14,
     x15,
     x16,
     x17,
     x18,
     x19,
     x20,
     x21,
     x22,
     x23,
     x24,
     x25,
     x26,
     x27,
     x28,
     x29,
     x30) = var(
        "x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30")

    x = [x1, x2, x3, x4, x5,
         x6, x7, x8, x9, x10,
         x11, x12, x13, x14,
         x15, x16, x17, x18,
         x19, x20, x21, x22,
         x23, x24, x25, x26,
         x27, x28, x29, x30]

    function = 30 * A

    for i in range(0, len(x)):
        function += (x[i] ** 2 - A * cos(2 * pi * x[i]))

    return x, function


def get_rastrigin30D_defs():
    
    X, function = __get_rastrigin_function()

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

