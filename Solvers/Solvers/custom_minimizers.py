__docformat__ = "restructuredtext en"

# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

# A collection of optimization algorithms. Version 0.5
# CHANGES
#  Added fminbound (July 2001)
#  Added brute (Aug. 2002)
#  Finished line search satisfying strong Wolfe conditions (Mar. 2004)
#  Updated strong Wolfe conditions line search to use
#  cubic-interpolation (Mar. 2004)

import warnings
import sys
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
                   asarray, sqrt, Inf, asfarray)
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._linesearch import (line_search_wolfe1,
                                        line_search_wolfe2,
                                        line_search_wolfe2 as line_search,
                                        LineSearchWarning)
from scipy.optimize._numdiff import approx_derivative
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                             'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}

#: LMA alpha computing parameters
_BETA = 2.0
_GAMMA = 3.0
_P = 3.0
_TAU = 0.1


def _indenter(s, n=0):
    """
    Ensures that lines after the first are indented by the specified amount
    """
    split = s.split("\n")
    indent = " " * n
    return ("\n" + indent).join(split)


def _float_formatter_10(x):
    """
    Returns a string representation of a float with exactly ten characters
    """
    if np.isposinf(x):
        return "       inf"
    elif np.isneginf(x):
        return "      -inf"
    elif np.isnan(x):
        return "       nan"
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)


def _dict_formatter(d, n=0, mplus=1, sorter=None):
    """
    Pretty printer for dictionaries
    `n` keeps track of the starting indentation;
    lines are indented by this much after a line break.
    `mplus` is additional left padding applied to keys
    """
    if isinstance(d, dict):
        m = max(map(len, list(d.keys()))) + mplus  # width to print keys
        s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                       _indenter(_dict_formatter(v, m + n + 2, 0, sorter), m + 2)
                       for k, v in sorter(d)])  # +2 for ': '
    else:
        # By default, NumPy arrays print with linewidth=76. `n` is
        # the indent at which a line begins printing, so it is subtracted
        # from the default to avoid exceeding 76 characters total.
        # `edgeitems` is the number of elements to include before and after
        # ellipses when arrays are not shown in full.
        # `threshold` is the maximum number of elements for which an
        # array is shown in full.
        # These values tend to work well for use with OptimizeResult.
        with np.printoptions(linewidth=76 - n, edgeitems=2, threshold=12,
                             formatter={'float_kind': _float_formatter_10}):
            s = str(d)
    return s


class OptimizeResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    Notes
    -----
    `OptimizeResult` may have additional attributes not listed here depending
    on the specific solver being used. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `OptimizeResult.keys` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        order_keys = ['message', 'success', 'status', 'fun', 'funl', 'x', 'xl',
                      'col_ind', 'nit', 'lower', 'upper', 'eqlin', 'ineqlin']
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {'slack', 'con', 'crossover_nit'}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class OptimizeWarning(UserWarning):
    pass


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)


def is_array_scalar(x):
    """Test whether `x` is either a scalar or an array scalar.
    """
    return np.size(x) == 1


_epsilon = sqrt(np.finfo(float).eps)


def vecnorm(x, ord=2):
    if ord == Inf:
        return np.amax(np.abs(x))
    elif ord == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x) ** ord, axis=0) ** (1.0 / ord)


def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    """
    Creates a ScalarFunction object for use with scalar minimizers
    (BFGS/LBFGSB/SLSQP/TNC/CG/etc).
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    jac : {callable,  '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:
            ``jac(x, *args) -> array_like, shape (n,)``
        If one of `{'2-point', '3-point', 'cs'}` is selected then the gradient
        is calculated with a relative step for finite differences. If `None`,
        then two-point finite differences with an absolute step is used.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` functions).
    bounds : sequence, optional
        Bounds on variables. 'new-style' bounds are required.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    hess : {callable,  '2-point', '3-point', 'cs', None}
        Computes the Hessian matrix. If it is callable, it should return the
        Hessian matrix:
            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
        Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
        finite difference scheme for numerical estimation.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.
    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        # default (jac is None) is to do 2-point finite differences with
        # absolute step size. ScalarFunction has to be provided an
        # epsilon value that is not None to use absolute steps. This is
        # normally the case from most _minimize* methods.
        grad = '2-point'
        epsilon = epsilon

    if hess is None:
        # ScalarFunction requires something for hess, so we give a dummy
        # implementation here if nothing is provided, return a value of None
        # so that downstream minimisers halt. The results of `fun.hess`
        # should not be used.
        def hess(x, *args):
            return None

    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction caches. Reuse of fun(x) during grad
    # calculation reduces overall function evaluations.
    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf


def _clip_x_for_func(func, bounds):
    # ensures that x values sent to func are clipped to bounds

    # this is used as a mitigation for gh11403, slsqp/tnc sometimes
    # suggest a move that is outside the limits by 1 or 2 ULP. This
    # unclean fix makes sure x is strictly within bounds.
    def eval(x):
        x = _check_clip_x(x, bounds)
        return func(x)

    return eval


def _check_clip_x(x, bounds):
    if (x < bounds[0]).any() or (x > bounds[1]).any():
        warnings.warn("Values in x were outside bounds during a "
                      "minimize step, clipping to bounds", RuntimeWarning)
        x = np.clip(x, bounds[0], bounds[1])
        return x

    return x


def _wrap_scalar_function(function, args):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        # Ideally, we'd like to a have a true scalar returned from f(x). For
        # backwards-compatibility, also allow np.array([1.3]), np.array([[1.3]]) etc.
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    return ncalls, function_wrapper


class _MaxFuncCallError(RuntimeError):
    pass


def _wrap_scalar_function_maxfun_validation(function, args, maxfun):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        if ncalls[0] >= maxfun:
            raise _MaxFuncCallError("Too many function calls")
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        # Ideally, we'd like to a have a true scalar returned from f(x). For
        # backwards-compatibility, also allow np.array([1.3]),
        # np.array([[1.3]]) etc.
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    return ncalls, function_wrapper


def approx_fprime(xk, f, epsilon=_epsilon, *args):
    """Finite difference approximation of the derivatives of a
    scalar or vector-valued function.
    If a function maps from :math:`R^n` to :math:`R^m`, its derivatives form
    an m-by-n matrix
    called the Jacobian, where an element :math:`(i, j)` is a partial
    derivative of f[i] with respect to ``xk[j]``.
    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the gradient of `f`.
    f : callable
        Function of which to estimate the derivatives of. Has the signature
        ``f(xk, *args)`` where `xk` is the argument in the form of a 1-D array
        and `args` is a tuple of any additional fixed parameters needed to
        completely specify the function. The argument `xk` passed to this
        function is an ndarray of shape (n,) (never a scalar even if n=1).
        It must return a 1-D array_like of shape (m,) or a scalar.
        .. versionchanged:: 1.9.0
            `f` is now able to return a 1-D array-like, with the :math:`(m, n)`
            Jacobian being estimated.
    epsilon : {float, array_like}, optional
        Increment to `xk` to use for determining the function gradient.
        If a scalar, uses the same finite difference delta for all partial
        derivatives. If an array, should contain one value per element of
        `xk`. Defaults to ``sqrt(np.finfo(float).eps)``, which is approximately
        1.49e-08.
    \\*args : args, optional
        Any other arguments that are to be passed to `f`.
    Returns
    -------
    jac : ndarray
        The partial derivatives of `f` to `xk`.
    See Also
    --------
    check_grad : Check correctness of gradient function against approx_fprime.
    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::
                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[i] = ---------------------------------
                            epsilon[i]
    Examples
    --------

    array([   2.        ,  400.00004198])
    """
    xk = np.asarray(xk, float)
    f0 = f(xk, *args)

    return approx_derivative(f, xk, method='2-point', abs_step=epsilon,
                             args=args, f0=f0)


def check_grad(func, grad, x0, *args, epsilon=_epsilon,
               direction='all', seed=None):
    """Check the correctness of a gradient function by comparing it against a
    (forward) finite-difference approximation of the gradient.
    Parameters
    ----------
    func : callable ``func(x0, *args)``
        Function whose derivative is to be checked.
    grad : callable ``grad(x0, *args)``
        Jacobian of `func`.
    x0 : ndarray
        Points to check `grad` against forward difference approximation of grad
        using `func`.
    args : \\*args, optional
        Extra arguments passed to `func` and `grad`.
    epsilon : float, optional
        Step size used for the finite difference approximation. It defaults to
        ``sqrt(np.finfo(float).eps)``, which is approximately 1.49e-08.
    direction : str, optional
        If set to ``'random'``, then gradients along a random vector
        are used to check `grad` against forward difference approximation
        using `func`. By default it is ``'all'``, in which case, all
        the one hot direction vectors are considered to check `grad`.
        If `func` is a vector valued function then only ``'all'`` can be used.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for reproducing the return value from this function.
        The random numbers generated with this seed affect the random vector
        along which gradients are computed to check ``grad``. Note that `seed`
        is only used when `direction` argument is set to `'random'`.
    Returns
    -------
    err : float
        The square root of the sum of squares (i.e., the 2-norm) of the
        difference between ``grad(x0, *args)`` and the finite difference
        approximation of `grad` using func at the points `x0`.
    See Also
    --------
    approx_fprime
    Examples
    --------

    """
    step = epsilon
    x0 = np.asarray(x0)

    def g(w, func, x0, v, *args):
        return func(x0 + w * v, *args)

    if direction == 'random':
        _grad = np.asanyarray(grad(x0, *args))
        if _grad.ndim > 1:
            raise ValueError("'random' can only be used with scalar valued"
                             " func")
        random_state = check_random_state(seed)
        v = random_state.normal(0, 1, size=(x0.shape))
        _args = (func, x0, v) + args
        _func = g
        vars = np.zeros((1,))
        analytical_grad = np.dot(_grad, v)
    elif direction == 'all':
        _args = args
        _func = func
        vars = x0
        analytical_grad = grad(x0, *args)
    else:
        raise ValueError("{} is not a valid string for "
                         "``direction`` argument".format(direction))

    return np.sqrt(np.sum(np.abs(
        (analytical_grad - approx_fprime(vars, _func, step, *_args)) ** 2
    )))


def approx_fhess_p(x0, p, fprime, epsilon, *args):
    # calculate fprime(x0) first, as this may be cached by ScalarFunction
    f1 = fprime(*((x0,) + args))
    f2 = fprime(*((x0 + epsilon * p,) + args))
    return (f2 - f1) / epsilon


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.
    Raises
    ------
    _LineSearchError
        If no suitable step size is found
    """

    extra_condition = kwargs.pop('extra_condition', None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def fmin_dfp(f, x0, fprime=None, args=(), gtol=1e-5, norm=Inf,
             epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
             retall=0, callback=None, xrtol=0):
    """
    Minimize a function using the DFP algorithm.
    Parameters
    ----------
    f : callable ``f(x,*args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable ``f'(x,*args)``, optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    gtol : float, optional
        Terminate successfully if gradient norm is less than `gtol`
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If `fprime` is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration. Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return ``fopt``, ``func_calls``, ``grad_calls``, and
        ``warnflag`` in addition to ``xopt``.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step
        size is less than ``xk * xrtol`` where ``xk`` is the current
        parameter vector.
    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e., ``f(xopt) == fopt``.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e., the inverse Hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
        3 : NaN result encountered.
    allvecs : list
        The value of `xopt` at each iteration. Only returned if `retall` is
        True.
    Notes
    -----
    Optimize the function, `f`, whose gradient is given by `fprime`
    using the quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS).
    See Also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See ``method='BFGS'`` in particular.
    References
    ----------
    Wright, and Nocedal 'Numerical Optimization', 1999, p. 198.
    Examples
    --------

    """
    opts = {'gtol': gtol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    res = _minimize_dfp(f, x0, args, fprime, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'],
                   res['nfev'], res['njev'], res['status'])
        if retall:
            retlist += (res['allvecs'],)
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_dfp(fun, x0, args=(), jac=None, callback=None,
                  gtol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                  disp=False, return_all=False, finite_diff_rel_step=None,
                  xrtol=0, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    DFP algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Terminate successfully if gradient norm is less than `gtol`.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step size is
        less than ``xk * xrtol`` where ``xk`` is the current parameter vector.
    """
    # _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    Bk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                     old_fval, old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        sk = alpha_k * pk
        xkp1 = xk + sk

        if retall:
            allvecs.append(xkp1)
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        #  See Chapter 5 in  P.E. Frandsen, K. Jonasson, H.B. Nielsen,
        #  O. Tingleff: "Unconstrained Optimization", IMM, DTU.  1999.
        #  These notes are available here:
        #  http://www2.imm.dtu.dk/documents/ftp/publlec.html
        if (alpha_k * vecnorm(pk) <= xrtol * (xrtol + vecnorm(xk))):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        # Cryptic comment above is preserved for posterity. Future reader:
        # consider change to condition below proposed in gh-1261/gh-17345.
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1. / rhok_inv

        '''A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])'''

        #: DFP Update: updating Bk and inverting for Hk

        A1 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        A2 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        Bk = np.dot(A1, np.dot(Bk, A2)) + (rhok * yk[:, np.newaxis] *
                                           yk[np.newaxis, :])
        Hk = np.linalg.inv(Bk)

    fval = old_fval

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def _minimize_lbfgs(fun,
                    x0,
                    args=(),
                    jac=None,
                    hess=None,
                    callback=None,
                    gtol=1e-5,
                    norm=Inf,
                    eps=_epsilon,
                    maxiter=None,
                    disp=False,
                    return_all=False,
                    finite_diff_rel_step=None,
                    xrtol=0,
                    **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    LBFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Terminate successfully if gradient norm is less than `gtol`.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step size is
        less than ``xk * xrtol`` where ``xk`` is the current parameter vector.
    """
    # _check_unknown_options(unknown_options)
    retall = return_all
    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 500

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    warnflag, allvecs, xk, k = __lbfgs(f, myfprime, x0, gtol, maxiter, 20)

    gfk = myfprime(xk)

    gnorm = vecnorm(gfk, ord=norm)

    fval = f(xk)

    Hk = hess(xk)

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def zoom(phi, phi_grad, alpha_lo, alpha_hi, c1, c2, max_iter=100):
    i = 0
    while True:
        alpha_j = (alpha_lo + alpha_hi) / 2.0
        phi_alpha_j = phi(alpha_j)

        if (phi_alpha_j > phi(0) + c1 * alpha_j * phi_grad(0)) or (phi_alpha_j >= phi(alpha_lo)):
            alpha_hi = alpha_j
        else:
            phi_grad_alpha_j = phi_grad(alpha_j)
            if np.abs(phi_grad_alpha_j) <= -c2 * phi_grad(0):
                return alpha_j
            if phi_grad_alpha_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
        i += 1
        if i >= max_iter:
            return None


def wolfe_line_search(f, grad, x, p, max_iter=100, c1=10 ** -4, c2=0.9, alpha_1=1.0, alpha_max=1000):
    def phi(alpha):
        return f(x + alpha * p)

    def phi_grad(alpha):
        return np.dot(grad(x + alpha * p).T, p)

    alpha_i_1 = 0
    alpha_i = alpha_1

    for i in range(1, max_iter + 1):
        phi_alpha_i = phi(alpha_i)
        if (phi_alpha_i > phi(0) + c1 * alpha_i * phi_grad(0)) or (i > 1 and phi_alpha_i >= phi(alpha_i_1)):
            return zoom(phi, phi_grad, alpha_i_1, alpha_i, c1, c2)

        phi_grad_alpha_i = phi_grad(alpha_i)
        if np.abs(phi_grad_alpha_i) <= -c2 * phi_grad(0):
            return alpha_i
        if phi_grad_alpha_i >= 0:
            return zoom(phi, phi_grad, alpha_i, alpha_i_1, c1, c2)
        alpha_i_1 = alpha_i
        alpha_i = min(2 * alpha_i, alpha_max)

    if i == max_iter:
        return None


def __lbfgs(f, grad, x0, eps=1e-4, max_iter=1000, history=10):
    n = len(x0)
    warnflag = 0
    k = 0

    def two_loop_rec(x, m):
        q = grad(x)
        k = S.shape[0]
        rhos = np.zeros(k)
        alphas = np.zeros(k)
        beta = 0
        for i in range(k - 1, -1, -1):
            rhos[i] = np.dot(Y[i].T, S[i]) ** -1
            alphas[i] = np.dot(S[i].T, q) * rhos[i]
            q = q - alphas[i] * Y[i]
        if k > 0:
            gamma_k = np.dot(S[k - 1].T, Y[k - 1]) / np.dot(Y[k - 1], Y[k - 1])
            H_k0 = np.diag(gamma_k * np.ones(n))
        else:
            H_k0 = np.diag(np.ones(n))
        r = np.dot(H_k0, q)
        for i in range(k):
            beta = rhos[i] * np.dot(Y[i].T, r)
            r = r + S[i] * (alphas[i] - beta)

        return r

    S = np.empty([0, n])
    Y = np.empty([0, n])
    x_old = x0
    allvecs = [x_old]

    for k in range(1, max_iter + 1):
        p_k = -two_loop_rec(x_old, history)
        alpha_k = wolfe_line_search(f, grad, x_old, p_k)

        if alpha_k is None:
            warnflag = 2
            return warnflag, allvecs, x_old, k

        x_new = x_old + alpha_k * p_k

        grad_diff = grad(x_new) - grad(x_old)

        if np.linalg.norm(grad_diff) < eps:
            break

        if k > history:
            S = S[1:]
            Y = Y[1:]

        S = np.append(S, [x_new - x_old], axis=0)

        Y = np.append(Y, [grad_diff], axis=0)

        x_old = x_new

        allvecs.append(x_old)

    '''if k == max_iter:
        print("Optimization did not converge")
    else:
        print("Optimization converged in {} steps".format(k))'''

    return warnflag, allvecs, x_old, k


"""
Computational Work 01 - Fundamentals of optimization
Authors: Augusto Mathias Adams - augusto.adams@ufpr.br - GRR20172143
         Caio Phillipe Mizerkowski - caiomizerkowski@gmail.com - GRR20166403
         Christian Piltz Araújo - christian0294@yahoo.com.br - GRR20172197
         Vinícius Eduardo dos Reis - eduardo.reis02@gmail.com - GRR20175957

lma helper functions

"""


def __compute_step(hessianF, gradF, alpha):
    """
    Computes the step increment of lma algorithm
    :param hessianF: the hessian matrix
    :param gradF: the gradient vector
    :param alpha: the step size
    :return: step update
    """

    step_update = -np.linalg.inv(hessianF + alpha * np.eye(len(hessianF))).dot(gradF.flatten())
    return step_update


def __compute_alpha(gain,
                    alpha,
                    nu):
    """
    Compute alpha - Nielsen, 2005
    :param gain: the gain of the iteration
    :param alpha: alpha value
    :param nu: velocity parameter
    :return: set the updated, alpha and nu new parameters
    """
    updated = False
    '''if gain > 0:
        alpha = np.max([1/_GAMMA, 1 - (_BETA - 1) * np.power(2 * gain - 1, _P)])
        nu = _BETA
        updated = True
    else:
        alpha *= nu
        nu *= 2'''
    if gain > 0:
        alpha *= 0.25
        updated = True
    else:
        alpha *= _BETA
    return updated, nu, alpha


def __compute_initial_alpha(gradF):
    '''ngradf = np.reshape(gradF, (len(gradF), 1))
    return _TAU * np.max(np.diag(ngradf.T.dot(ngradf)))'''
    return 100


def compute_stop_criteria(oldF, newF):
    return np.abs(oldF - newF)


def __hessianF(gfk):

    ngfk = np.reshape(gfk, (1, len(gfk)))
    return ngfk.T.dot(ngfk)


def _minimize_lma(fun,
                  x0,
                  args=(),
                  jac=None,
                  hess=None,
                  callback=None,
                  gtol=1e-5,
                  norm=Inf,
                  eps=_epsilon,
                  maxiter=None,
                  disp=False,
                  return_all=False,
                  finite_diff_rel_step=None,
                  xrtol=0,
                  **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    LMA algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Terminate successfully if gradient norm is less than `gtol`.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    xrtol : float, default: 0
        Relative tolerance for `x`. Terminate successfully if step size is
        less than ``xk * xrtol`` where ``xk`` is the current parameter vector.
    """
    # _check_unknown_options(unknown_options)
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = 4000

    sf = _prepare_scalar_function(fun,
                                  x0,
                                  jac,
                                  args=args,
                                  epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  hess=hess)

    f = sf.fun
    myfprime = sf.grad
    mysprime = sf.hess

    old_fval = f(x0)
    gfk = myfprime(x0)
    hessianF = mysprime(x0)
    alpha = __compute_initial_alpha(gfk)
    nu = _BETA
    xk = x0
    k = 0
    if retall:
        allvecs = [x0]

    warnflag = 0

    gnorm = 1

    while (gnorm > gtol) and (k < maxiter):

        with warnings.catch_warnings():

            warnings.filterwarnings('error')

            try:
                # step update - RAO 2019
                lbda = __compute_step(hessianF, gfk, alpha)

                # gain - Nielsen, 2005

                txk = xk + lbda

                newfk = f(txk)

                newgfk = myfprime(txk)

                gain = old_fval - newfk

                update, alpha, nu = __compute_alpha(gain, alpha, nu)

                if update:

                    xk = txk

                    old_fval = newfk

                    k += 1

                    gfk = newgfk

                    hessianF = mysprime(xk)

                    if retall:
                        allvecs.append(xk)

                    if callback is not None:
                        callback(xk)

                    gnorm = vecnorm(gfk, ord=norm)

                    if (gnorm <= gtol):
                        break

            except Warning:
                warnflag = 4
                break

    fval = f(xk)

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    elif warnflag == 4:
        msg = "LingAlg Error"
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Hessian evaluations: %d" % sf.nhev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=hessianF, nfev=sf.nfev,
                            njev=sf.ngev, nhev=sf.nhev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result
