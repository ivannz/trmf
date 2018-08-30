"""Trust region optimizer."""
from math import sqrt
import numpy as np


def trcg(Ax, r, x, n_iterations=1000, tr_delta=0, rtol=1e-5, atol=1e-8,
         args=(), verbose=False):
    r"""Simple Conjugate gradient solver with trust region control.

    Approximately solves `r = A(z - x)` w.r.t. `z` and updating `r` and `x`
    inplace with the final residual and solution `z`, respectively.

    Details
    -------
    For the given `r` this procedure uses conjugate gradients method to solve
    the least squares problem within the trust region of radius :math:`\delta`
    around `x`:

        .. math ::
            \min_{p} \|A p - r \|^2
            s.t. \|x + p\| \leq \delta

    and returns `z = x + p` as the solution. The residual `r` and the point `x`
    are updated inpalce with the final residual and solution `z`, respectively,
    upon termination.

    Backtracking
    ------------
    In contrast to the implementation on LIBLINEAR, this version backtrack into
    the trust region upon a breach event. It does using the analytic solution
    to the following 1-dimensional optimisation problem::

        .. math ::
            \min_{\eta \geq 0} \eta
            s.t. \| (z - x) - \eta p \| \leq \delta

    where :math:`\|z - x\| > \delta` and `p` is the current conjugate
    minimizing direction. The solution is given by

        .. math ::
            \eta = \frac1{\|p\|} (q - \sqrt{q^2 + \delta^2 - \|z-x\|^2})

    where :math:`q = \frac{p'(z-x)}{\|p\|}`.

    Arguments
    ---------
    Ax : callable
        The function with declaration `f(x, *args)` that computes the matrix
        vector product for the given point `x`.

    r : flat writable numpy array
        The initial residual vector to solve the linear system for. The array
        in `r` is updated INPLACE during the iterations. The final solution
        residual is returned implicitly in `r`.

    x : flat writable numpy array
        The initial point and final solution of the linear system. The array
        in `x` is updated INPLACE during the iterations. The solution is
        returned implicitly in `x`.

    n_iterations : int, optional (default=1000)
        The number of tron iterations.

    tr_delta : double, optional (default=0)
        The radius of the trust region around the initial point in `x`. The
        conjugate gradient steps steps are terminated if the trust region is
        breached, in which case the constraint violating step is retracted
        back to the trust region boundary.

    rtol : double, optional (default=1e-3)
        The relative reduction of the l2 norm of the gradient, to be used in
        for convergence criterion. The default set to match the corresponding
        setting in LIBLINEAR.

    atol : double, optional (default=1e-5)
        The minimal absolute reduction in the l2 norm of the gradient.

    args : tuple, optional (default=empty tuple)
        The extra positional arguments to pass the the callables in `func`.

    verbose : bool, optional (default=False)
        Whether to print debug diagnostics regarding the convergence, the
        current trust region radius, gradients, CG iterations and step sizes.


    Examples
    --------

    Import numpy and trcg from this library

    >>> import numpy as np
    >>> from trmf.tron import trcg

    Create radon psd matrix A

    >>> A = np.random.normal(scale=0.1, size=(10000, 200))
    >>> A = np.dot(A.T, A)

    Solve using linalv.inv (pivoting)

    >>> r_0, a_0 = np.ones(200), np.ones(200)
    >>> z_0 = a_0 + np.linalg.solve(A, r_0)

    Solve using trcg

    >>> r, z = r_0.copy(), a_0.copy()
    >>> trcg(lambda p: np.dot(A, p), r, z, verbose=False)
    >>> assert np.allclose(z, z_0)

    """
    if n_iterations > 0:
        n_iterations = min(n_iterations, len(x))

    p, iteration = r.copy(), 0
    tr_delta_sq = tr_delta ** 2

    rtr, rtr_old = np.dot(r, r), 1.0
    cg_tol = sqrt(rtr) * rtol + atol
    region_breached = False
    while (iteration < n_iterations) and (sqrt(rtr) > cg_tol):
        Ap = Ax(p, *args)
        iteration += 1
        if verbose:
            print("""iter %2d |Ap| %5.3e |p| %5.3e """
                  """|r| %5.3e |x| %5.3e beta %5.3e""" %
                  (iteration, np.linalg.norm(Ap), np.linalg.norm(p),
                   np.linalg.norm(r), np.linalg.norm(x), rtr / rtr_old))
        # end if

        # ddot(&n, p, &inc, Ap, &inc);
        alpha = rtr / np.dot(p, Ap)
        # daxpy(&n, &alpha, p, &inc, x, &inc);
        x += alpha * p
        # daxpy(&n, &( -alpha ), Ap, &inc, r, &inc);
        r -= alpha * Ap

        # check trust region (diverges from tron.cpp in liblinear and leml-imf)
        if tr_delta_sq > 0:
            xTx = np.dot(x, x)
            if xTx > tr_delta_sq:
                xTp = np.dot(x, p)
                if xTp > 0:
                    # backtrack into the trust region
                    p_nrm = np.linalg.norm(p)

                    q = xTp / p_nrm
                    eta = (q - sqrt(max(q * q + tr_delta_sq - xTx, 0))) / p_nrm

                    # reproject onto the boundary of the region
                    r += eta * Ap
                    x -= eta * p
                else:
                    # this never happens maybe due to CG iteration properties
                    pass
                # end if

                region_breached = True
                break
            # end if
        # end if

        # ddot(&n, r, &inc, r, &inc);
        rtr, rtr_old = np.dot(r, r), rtr
        # dscal(&n, &(rtr / rtr_old), p, &inc);
        p *= rtr / rtr_old
        # daxpy(&n, &one, r, &1, p, &1);
        p += r
    # end while

    return iteration, region_breached


def tron(func, x, n_iterations=1000, rtol=1e-3, atol=1e-5, args=(),
         verbose=False):
    """Trust Region Newton optimisation.

    This method minimizes the given objective function by adaptively choosing
    the radius of the trust region, within which a quadratic approximation of
    the  function is trustworthy.

    This routine implements the TRON method from [1]_ and [2]_, and is based
    on the C++ implementation in LIBLINEAR, [3]_.

    Arguments
    ---------
    func : tuple
        A tuple consisting of 3 callables with declaration `f(x, *args)`
        1. `f_valp` -- returns the current value of the objective, which
            is allowed to affect the internal state.
        2. `f_grad` -- returns the gradient at the point of the most recently
            evaluated objective.
        3. `f_hess` -- returns the hessian-vector product at the point of the
            most recent evaluation of the objective.
        The implementation requires that neither `f_grad`, nor `f_hess` affect
        the internal state of the minimized function, i.e. 2nd order Taylor
        series approximation. It is guaranteed that `f_grad` and `f_hess` are
        called only after evaluation `f_valp`.
        These callables are typically methods of an instance of an quadratic
        approximation class for a double smooth function.

    x : flat writable numpy array
        The starting point and the solution of the tron iterations. The data in
        `x` is updated INPLACE during the iterations. The terminal point is
        returned implicitly in `x`.

    n_iterations : int, optional (default=1000)
        The number of tron iterations.

    rtol : double, optional (default=1e-3)
        The relative reduction of the l2 norm of the gradient, to be used in
        for convergence criterion. The default set to match the corresponding
        setting in LIBLINEAR.

    atol : double, optional (default=1e-5)
        The minimal absolute reduction in the l2 norm of the gradient.

    args : tuple, optional (default=empty tuple)
        The extra positional arguments to pass the the callables in `func`.

    verbose : bool, optional (default=False)
        Whether to print debug diagnostics regarding the convergence, the
        current trust region radius, gradients, CG iterations and step sizes.

    Returns
    -------
    cg_iter : int
        The total number of Conjugate gradient iterations.

    References
    ----------
    .. [1] Hsia, C. Y., Zhu, Y., & Lin, C. J. (2017, November). "A study on
           trust region update rules in Newton methods for large-scale linear
           classification." In Asian Conference on Machine Learning
           (pp. 33-48).

    .. [2] Lin, C. J., Weng, R. C., & Keerthi, S. S. (2008). "Trust region
           newton method for logistic regression." Journal of Machine Learning
           Research, 9(Apr), 627-650.

    .. [3] Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J.,
           (2008). "LIBLINEAR: A Library for Large Linear Classification",
           Journal of Machine Learning Research 9, 1871-1874.
           Software available at http://www.csie.ntu.edu.tw/~cjlin/liblinear
    """
    eta0, eta1, eta2 = 1e-4, 0.25, 0.75
    sigma1, sigma2, sigma3 = 0.25, 0.5, 4.0

    f_valp_, f_grad_, f_hess_ = func

    iteration, cg_iter = 0, 0

    fval = f_valp_(x, *args)
    grad = f_grad_(x, *args)
    grad_norm = np.linalg.norm(grad)

    # make a copy of `-grad` and zeros like `x`
    # r, z = -grad, np.zeros_like(x)
    delta, grad_norm_tol = grad_norm, grad_norm * rtol + atol
    while iteration < n_iterations and grad_norm > grad_norm_tol:
        r, z = -grad, np.zeros_like(x)
        # tolerances and n_iterations as in leml-imf
        cg_iter, region_breached = trcg(
            f_hess_, r, z, tr_delta=delta, args=args,
            n_iterations=20, rtol=1e-1, atol=0.0)

        z_norm = np.linalg.norm(z)
        if iteration == 0:
            delta = min(delta, z_norm)

        # trcg finds z and r s.t. r + A z = -g and \|r\|\to \min
        # f(x) - f(x+z) ~ -0.5 * (2 g'z + z'Az) = -0.5 * (g'z + z'(-r))
        linear = np.dot(z, grad)
        approxred = -0.5 * (linear - np.dot(z, r))

        # The value and the actual reduction: compute the forward pass.
        fnew = f_valp_(x + z, *args)
        actualred = fval - fnew

        if linear + actualred < 0:
            alpha = max(sigma1, 0.5 * linear / (linear + actualred))

        else:
            alpha = sigma3

        # end if

        if actualred < eta0 * approxred:
            delta = min(max(alpha, sigma1) * z_norm, sigma2 * delta)

        elif actualred < eta1 * approxred:
            delta = max(sigma1 * delta, min(alpha * z_norm, sigma2 * delta))

        elif actualred < eta2 * approxred:
            delta = max(sigma1 * delta, min(alpha * z_norm, sigma3 * delta))

        else:
            # patch 2018-08-30: new addition from tron.cpp at
            #  https://github.com/cjlin1/liblinear/blob/master/tron.cpp
            if region_breached:
                delta = sigma3 * delta

            else:
                delta = max(delta, min(alpha * z_norm, sigma3 * delta))

        # end if

        if verbose:
            print("""iter %2d act %5.3e pre %5.3e delta %5.3e """
                  """f %5.3e |z| %5.3e |g| %5.3e CG %3d""" %
                  (iteration, actualred, approxred,
                   delta, fval, z_norm, grad_norm, cg_iter))
        # end if

        if actualred > eta0 * approxred:
            x += z
            fval, grad = fnew, f_grad_(x, *args)
            grad_norm = np.linalg.norm(grad)
            iteration += 1

            # r, z = -grad, np.zeros_like(x)
        # end if

        if fval < -1e32:
            if verbose:
                print("WARNING: f < -1.0e+32")
            break
        # end if

        if abs(actualred) <= 0 and approxred <= 0:
            if verbose:
                print("WARNING: actred and prered <= 0")
            break
        # end if

        if abs(actualred) <= 1e-12 * abs(fval) and \
           abs(approxred) <= 1e-12 * abs(fval):
            if verbose:
                print("WARNING: actred and prered too small")
            break
        # end if

        if delta <= rtol * (z_norm + atol):
            if verbose:
                print("WARNING: degenerate trust region")
            break
        # end if
    # end while

    return cg_iter
