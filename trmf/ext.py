# Extensions: not included in the prototype

import numpy as np
import numba as nb

import scipy.sparse as sp

from .tron import tron

from sklearn.utils.extmath import safe_sparse_dot
from .extmath import csr_gemm
from .steps import l2_loss_valj


def b_step_tron_valj(b, Y, X, C_B):
    """Compute current value the b-step loss."""
    (n_samples, n_targets), n_features = Y.shape, X.shape[1]

    B = b.reshape(n_features, n_targets)
    objective = l2_loss_valj(Y, X, B)

    if sp.issparse(Y):
        coef = C_B * Y.nnz / (n_features * n_targets)
    else:
        coef = C_B * n_samples / n_features

    if C_B > 0:
        reg_b = np.linalg.norm(B, ord="fro") ** 2

        objective += reg_b * coef
    # end if

    return 0.5 * objective


def b_step_tron_grad(b, Y, X, C_B):
    """Compute the gradient of the b-step objective."""
    (n_samples, n_targets), n_features = Y.shape, X.shape[1]

    B = b.reshape(n_features, n_targets)
    if sp.issparse(Y):
        coef = C_B * Y.nnz / (n_features * n_targets)

        grad = safe_sparse_dot(X.T, csr_gemm(1, X, B, -1, Y.copy()))
        grad += coef * B

    else:
        coef = C_B * n_samples / n_features

        XTY, XTX = np.dot(X.T, Y), np.dot(X.T, X)
        if C_B > 0:
            XTX.flat[::n_features + 1] += coef

        grad = np.dot(XTX, B) - XTY
    # end if

    return grad.reshape(-1)


def b_step_tron_hess(v, Y, X, C_B):
    """Get the Hessian-vector product for the b-step objective."""
    (n_samples, n_targets), n_features = Y.shape, X.shape[1]

    V = v.reshape(n_features, n_targets)
    if sp.issparse(Y):
        coef = C_B * Y.nnz / (n_features * n_targets)

        hess_v = safe_sparse_dot(X.T, csr_gemm(1, X, V, 0, Y.copy()))
        hess_v += coef * V

    else:
        coef = C_B * n_samples / n_features

        XTX = np.dot(X.T, X)
        if C_B > 0:
            XTX.flat[::n_features + 1] += coef

        hess_v = np.dot(XTX, V)
    # end if

    return hess_v.reshape(-1)


def b_step_tron(B, Y, X, C_B, rtol=5e-2, atol=1e-4, verbose=False, **kwargs):
    """TRON solver for the b-step minimization problem."""
    f_call = b_step_tron_valj, b_step_tron_grad, b_step_tron_hess

    tron(f_call, B.ravel(), n_iterations=5, rtol=rtol, atol=atol,
         args=(Y, X, C_B), verbose=verbose)

    return B


def soft_prox(x, c):
    return np.maximum(x - c, 0.) + np.minimum(x + c, 0.)


# def b_step_prox(B, Y, X, C_B, lip=1e-2, n_iter=25, alpha=1.0, **kwargs):
#     gamma_u, gamma_d = 2, 1.1

#     # get the gradient
#     grad = b_step_prox_grad(B, Y, X, C_B)
#     grad_F = np.dot(grad.flat, F.flat)

#     f0, lip0 = b_step_prox_func(B, Y, X, C_B), lip
#     for _ in range(n_iter):
#         # F_new = (1 -  alpha) * F + alpha * np.maximum(F - lr * grad, 0.)
#         # prox-sgd operation
#         F_new = np.maximum(F - grad / lip, 0.)

#         # fgm lipschitz search
#         delta = b_step_prox_func(F_new, Y, X, C_F, eta_F, adj) - f0
#         linear = np.dot(grad.flat, F_new.flat) - grad_F
#         quad = np.linalg.norm(F_new - F, ord="fro") ** 2
#         if delta <= linear + lip * quad / 2:
#             break
#         lip *= gamma_u
#     # end for
# #     lip = max(lip0, lip / gamma_d)
#     lip = lip / gamma_d

#     return F_new, lip


def b_step(B, Y, X, C_B, kind="tron", **kwargs):
    """A common subroutine solving the b-step minimization problem."""
    lip = np.inf
    if kind == "tron":
        B = b_step_tron(B, Y, X, C_B, **kwargs)
    # elif kind == "fgm":
    #     B, lip = b_step_prox(B, Y, X, C_B, **kwargs)
    else:
        raise ValueError(f"""Unrecognozed optimization `{kind}`""")

    return B, lip
