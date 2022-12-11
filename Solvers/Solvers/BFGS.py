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


def bfgs(of, jac, jac2, x0):
    """
    Minimizes a funtion using Quasi-Newton BFGS method
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
                      method='BFGS',
                      jac=derivative,
                      tol=1e-9,
                      options={'return_all': True})

    return result