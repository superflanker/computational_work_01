"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

Rosenbrock 30D function and its derivatives
"""

from sympy import var
from .common import *


def __get_rosenbrock_function():
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
    x30) = var("x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30")

    x = [x1, x2, x3, x4, x5,
         x6, x7, x8, x9, x10,
         x11, x12, x13, x14,
         x15, x16, x17, x18,
         x19, x20, x21, x22,
         x23, x24, x25, x26,
         x27, x28, x29, x30]

    function = 100 * ((x[1] - x[0] ** 2) ** 2) + (x[0] - 1) ** 2

    for i in range(1, len(x)-1):

         function += 100 * ((x[i+1] - x[i] ** 2) ** 2) + (x[i] - 1) ** 2

    return x, function


def get_rosenbrock30D_defs():

    X, function = __get_rosenbrock_function()

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
