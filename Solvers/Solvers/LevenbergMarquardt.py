"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

LMA algorithm
The minimizer is adapted from the original SciPy BFGS implementation, since DFP
is it's dual algorithm
"""

import numpy as np
from scipy.optimize import minimize
from ..common import *
from .custom_minimizers import _minimize_lma


def lma(of, jac, jac2, x0):
    """
    Minimizes a funtion using Levenberg-Marquardt Algorithm
    :param f: the function to be minimized
    :param jac: the analytical jacobian (gradient) function
    :param x0: the initial guess
    :return: optimization results
    """
    objective = lambda x: of(*x.flatten())

    derivative = lambda x: jac(*x.flatten())

    hessian = lambda x: jac2(*x.flatten())

    result = minimize(objective,
                      x0,
                      jac=derivative,
                      hess=hessian,
                      tol=1e-11,
                      method=_minimize_lma,
                      options={'return_all': True})

    return result
