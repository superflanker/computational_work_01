"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

BGFS algorithm
"""
import numpy as np
from ..common import *
from scipy.optimize import minimize
from .custom_minimizers import _minimize_lbfgs


def lbfgs(of, jac, jac2, x0):
    """
    Minimizes a funtion using Quasi-Newton LBFGS method
    :param of: the function to be minimized
    :param jac: the analytical jacobian (gradient) function
    :param x0: the initial guess
    :return: optimization results
    """
    objective = lambda x: of(*x.flatten())

    derivative = lambda x: jac(*x.flatten())

    hessian = lambda x: jac2(*x.flatten())

    result = minimize(objective,
                      x0,
                      method=_minimize_lbfgs,
                      jac=derivative,
                      hess=hessian, # only to compute final hessian values - to be compatible with other solvers
                      tol=1e-9,
                      options={'return_all': True})
    return result