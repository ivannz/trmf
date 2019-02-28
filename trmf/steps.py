# updated from version of `trmf_v0.9.ipynb` as of 2019-02-28 17:00

import numpy as np
import numba as nb

import scipy.sparse as sp

from scipy.optimize import fmin_l_bfgs_b, fmin_ncg

from numpy.lib.stride_tricks import as_strided
from sklearn.utils.extmath import safe_sparse_dot

from .tron import tron
from .extmath import csr_gemm


# In[13]:
@nb.njit("float64[:,::1](float64[:,::1], float64[:,::1])",
         fastmath=True, cache=False, error_model="numpy")
def ar_resid(Z, phi):
    """Compute the AR(p) residuals of the multivariate data in Z."""
    n_components, n_order = phi.shape

    resid = Z[n_order:].copy()
    for k in range(n_order):
        # r_t -= y_{t-(p-k)} * \beta_{p - k} (\phi is reversed \beta)
        resid -= Z[k:k - n_order] * phi[:, k]

    return resid


# In[15]:
@nb.njit("float64[:,::1](float64[:,::1], float64[:,::1], float64[:,::1])",
         fastmath=True, cache=False, error_model="numpy")
def ar_hess_vect(V, Z, phi):
    """Compute the Hessian-vector product of the AR(p) square loss for `V`."""
    n_components, n_order = phi.shape

    # compute the AR(p) residuals over V
    resid = ar_resid(V, phi)

    # get the derivative w.r.t. the series
    hess_v = np.zeros_like(V)
    hess_v[n_order:] = resid
    for k in range(n_order):
        hess_v[k:k - n_order] -= resid * phi[:, k]

    return hess_v


# In[16]:
@nb.njit("float64[:,::1](float64[:,::1], float64[:,::1])",
         fastmath=True, cache=False, error_model="numpy")
def ar_grad(Z, phi):
    """Compute the gradient of the AR(p) l2 loss w.r.t. the time-series `Z`."""
    return ar_hess_vect(Z, Z, phi)


def precompute_graph_reg(adj):
    """Precompute the neighbor average discrepancy operator."""

    # make a copy of the adjacency matrix and the outbound degree
    resid, deg = adj.astype(float), adj.getnnz(axis=1)

    # scale the rows : D^{-1} A
    resid.data /= deg[adj.nonzero()[0]]

    # subtract the matrix from the diagonalized mask
    return sp.diags((deg > 0).astype(float)) - resid


def graph_resid(F, adj):
    """Get the residual of the outgoing neighbor average of `F`."""
    return safe_sparse_dot(adj, F.T).T


def graph_grad(F, adj):
    """Compute the gradient of the outgoing neighbors average w.r.t. `F`."""
    return safe_sparse_dot(adj.T, graph_resid(F, adj).T).T


def graph_hess_vect(V, F, adj):
    """Get the Hessian-vector product of the outgoing neighbors average."""
    return graph_grad(V, adj)


# In[28] -- sparse:
def l2_loss_valj(Y, Z, F):
    if sp.issparse(Y):
        R = csr_gemm(1, Z, F, -1, Y.copy())
        return sp.linalg.norm(R, ord="fro") ** 2

    return np.linalg.norm(Y - np.dot(Z, F), ord="fro") ** 2


# In[31]:
def f_step_tron_valj(f, Y, Z, C_F, eta_F, adj):
    """Compute current value the f-step loss."""
    (n_samples, n_targets), n_components = Y.shape, Z.shape[1]

    F = f.reshape(n_components, n_targets)
    objective = l2_loss_valj(Y, Z, F)

    if sp.issparse(Y):
        coef = C_F * Y.nnz / (n_components * n_targets)
    else:
        coef = C_F * n_samples / n_components

    if C_F > 0:
        if eta_F < 1:
            reg_f_l2 = np.linalg.norm(F, ord="fro") ** 2
        else:
            reg_f_l2, eta_F = 0., 1.
        # end if

        if sp.issparse(adj) and (eta_F > 0):
            reg_f_graph = np.linalg.norm(graph_resid(F, adj), ord="fro") ** 2
        else:
            reg_f_graph, eta_F = 0., 0.
        # end if

        reg_f = reg_f_l2 * (1 - eta_F) + reg_f_graph * eta_F
        objective += reg_f * coef
    # end if

    return 0.5 * objective


# In[33]:
def f_step_tron_grad(f, Y, Z, C_F, eta_F, adj):
    """Compute the gradient of the f-step objective."""
    (n_samples, n_targets), n_components = Y.shape, Z.shape[1]

    F = f.reshape(n_components, n_targets)
    if sp.issparse(Y):
        coef = C_F * Y.nnz / (n_components * n_targets)

        grad = safe_sparse_dot(Z.T, csr_gemm(1, Z, F, -1, Y.copy()))
        grad += (1 - eta_F) * coef * F

    else:
        coef = C_F * n_samples / n_components

        ZTY, ZTZ = np.dot(Z.T, Y), np.dot(Z.T, Z)
        if C_F > 0 and eta_F < 1:
            ZTZ.flat[::n_components + 1] += (1 - eta_F) * coef

        grad = np.dot(ZTZ, F) - ZTY
    # end if

    if C_F > 0 and sp.issparse(adj) and eta_F > 0:
        grad += graph_grad(F, adj) * (eta_F * coef)

    return grad.reshape(-1)


# In[34]:
def f_step_tron_hess(v, Y, Z, C_F, eta_F, adj):
    """Get the Hessian-vector product for the f-step objective."""
    (n_samples, n_targets), n_components = Y.shape, Z.shape[1]

    V = v.reshape(n_components, n_targets)
    if sp.issparse(Y):
        coef = C_F * Y.nnz / (n_components * n_targets)

        hess_v = safe_sparse_dot(Z.T, csr_gemm(1, Z, V, 0, Y.copy()))
        hess_v += (1 - eta_F) * coef * V

    else:
        coef = C_F * n_samples / n_components

        ZTZ = np.dot(Z.T, Z)
        if C_F > 0 and eta_F < 1:
            ZTZ.flat[::n_components + 1] += (1 - eta_F) * coef

        hess_v = np.dot(ZTZ, V)
    # end if

    if C_F > 0 and sp.issparse(adj) and eta_F > 0:
        hess_v += graph_grad(V, adj) * (eta_F * coef)

    return hess_v.reshape(-1)


# In[32]:
def f_step_tron(F, Y, Z, C_F, eta_F, adj, rtol=5e-2, atol=1e-4, verbose=False,
                **kwargs):
    """TRON solver for the f-step minimization problem."""
    f_call = f_step_tron_valj, f_step_tron_grad, f_step_tron_hess

    tron(f_call, F.ravel(), n_iterations=5, rtol=rtol, atol=atol,
         args=(Y, Z, C_F, eta_F, adj), verbose=verbose)

    return F


def f_step_ncg_hess_(F, v, Y, Z, C_F, eta_F, adj):
    """A wrapper of the hess-vector product for ncg calls."""
    return f_step_tron_hess(v, Y, Z, C_F, eta_F, adj)


def f_step_ncg(F, Y, Z, C_F, eta_F, adj, **kwargs):
    """Solve the F-step using scipy's Newton-CG."""
    FF = fmin_ncg(f=f_step_tron_valj, x0=F.ravel(), disp=False,
                  fprime=f_step_tron_grad, fhess_p=f_step_ncg_hess_,
                  args=(Y, Z, C_F, eta_F, adj))

    return FF.reshape(F.shape)


def f_step_lbfgs(F, Y, Z, C_F, eta_F, adj, **kwargs):
    """Solve the F-step using scipy's L-BFGS method."""
    FF, f, d = fmin_l_bfgs_b(func=f_step_tron_valj, x0=F.ravel(), iprint=0,
                             fprime=f_step_tron_grad,
                             args=(Y, Z, C_F, eta_F, adj))

    return FF.reshape(F.shape)


# In[33]:
def f_step_prox_func(F, Y, Z, C_F, eta_F, adj):
    """An interface to the f-step objective for unraveled matrices."""
    return f_step_tron_valj(F.ravel(), Y, Z, C_F, eta_F, adj)


# In[34]:
def f_step_prox_grad(F, Y, Z, C_F, eta_F, adj):
    """An interface to the f-step objective gradient for unraveled matrices."""
    return f_step_tron_grad(F.ravel(), Y, Z, C_F, eta_F, adj).reshape(F.shape)


# In[35]:
def f_step_prox(F, Y, Z, C_F, eta_F, adj, lip=1e-2, n_iter=25, alpha=1.0,
                **kwargs):
    """Use Fast Prox gradient Method to solve the constrained f-step problem.

    The fast proximal Gradient Methods controls the current discrepancy
    between the strongly convex majorant and support to select the step size
    based on adaptive estimation of the Lischitz constant.

    Arguments
    ---------

    Returns
    -------

    Details
    -------
    """
    gamma_u, gamma_d = 2, 1.1

    # get the gradient
    grad = f_step_prox_grad(F, Y, Z, C_F, eta_F, adj)
    grad_F = np.dot(grad.flat, F.flat)

    f0, lip0 = f_step_prox_func(F, Y, Z, C_F, eta_F, adj), lip
    for _ in range(n_iter):
        # F_new = (1 -  alpha) * F + alpha * np.maximum(F - lr * grad, 0.)
        # prox-sgd operation
        F_new = np.maximum(F - grad / lip, 0.)

        # FGM Lipschitz search
        delta = f_step_prox_func(F_new, Y, Z, C_F, eta_F, adj) - f0
        linear = np.dot(grad.flat, F_new.flat) - grad_F
        quad = np.linalg.norm(F_new - F, ord="fro") ** 2
        if delta <= linear + lip * quad / 2:
            break

        lip *= gamma_u
    # end for

    # lip = max(lip0, lip / gamma_d)
    lip = lip / gamma_d

    return F_new, lip


# In[37]:
def f_step(F, Y, Z, C_F, eta_F, adj, kind="fgm", **kwargs):
    """A common subroutine solving the f-step minimization problem."""
    lip = np.inf
    if kind == "fgm":
        F, lip = f_step_prox(F, Y, Z, C_F, eta_F, adj, **kwargs)
    elif kind == "tron":
        F = f_step_tron(F, Y, Z, C_F, eta_F, adj, **kwargs)
    elif kind == "ncg":
        F = f_step_ncg(F, Y, Z, C_F, eta_F, adj, **kwargs)
    elif kind == "lbfgs":
        F = f_step_lbfgs(F, Y, Z, C_F, eta_F, adj, **kwargs)
    else:
        raise ValueError(f"""Unrecognized method `{kind}`""")

    return F, lip


# In[41]:
def phi_step(phi, Z, C_Z, C_phi, eta_Z, nugget=1e-8):
    """Compute the Ridge regression estimate of the AR(p) coefficients.

    The assumed model of the latent autoregressive factors is diagonal VAR(p),
    which is a constrained variant of the general vector autoregression of
    order `p`.
    """
    # return a set of independent AR(p) ridge estimates.
    (n_components, n_order), n_samples = phi.shape, Z.shape[0]
    if n_order < 1 or n_components < 1:
        return np.empty((n_components, n_order))

    if not (C_Z > 0 and eta_Z > 0):
        return np.zeros_like(phi)

    # embed into the last dimensions
    shape = Z.shape[1:] + (Z.shape[0] - n_order, n_order + 1)
    strides = Z.strides[1:] + Z.strides[:1] + Z.strides[:1]
    Z_view = as_strided(Z, shape=shape, strides=strides)

    # split into y (d x T-p) and Z (d x T-p x p) (all are views!)
    y, Z_lagged = Z_view[..., -1], Z_view[..., :-1]

    # compute the SVD: thin, but V is d x p x p
    U, s, Vh = np.linalg.svd(Z_lagged, full_matrices=False)
    if C_phi > 0:
        # the {V^{H}}^{H} (\Sigma^2 + C I)^{-1} \Sigma part is reduced
        #  to columnwise operations
        gain = (C_Z * eta_Z * n_order) * s
        gain /= gain * s + C_phi * (n_samples - n_order)
    else:
        # do the same cutoff as in np.linalg.pinv(...)
        large = s > nugget * np.max(s, axis=-1, keepdims=True)
        gain = np.divide(1, s, where=large, out=s)
        gain[~large] = 0
    # end if

    # get the U' y part and the final estimate
    # $\phi_j$ corresponds to $p-j$-th lag $j = 0,\,\ldots,\,p-1$
    return np.einsum("ijk,ij,isj,is->ik", Vh, gain, U, y)


# In[46]:
def z_step_tron_valh(z, Y, F, phi, C_Z, eta_Z):
    """Get the value of the z-step objective."""
    n_samples, n_targets = Y.shape
    n_components, n_order = phi.shape

    Z = z.reshape(n_samples, n_components)
    objective = l2_loss_valj(Y, Z, F)

    if sp.issparse(Y):
        coef = C_Z * Y.nnz / (n_samples * n_components)
    else:
        coef = C_Z * n_targets / n_components

    if C_Z > 0:
        if eta_Z < 1:
            reg_z_l2 = np.linalg.norm(Z, ord="fro") ** 2
        else:
            reg_z_l2, eta_Z = 0., 1.
        # end if

        if eta_Z > 0 and n_samples > n_order:
            reg_z_ar_j = np.linalg.norm(ar_resid(Z, phi), ord=2, axis=0) ** 2
            reg_z_ar = np.sum(reg_z_ar_j) * n_samples / (n_samples - n_order)
        else:
            reg_z_ar, eta_Z = 0., 0.
        # end if

        # reg_z was implicitly scaled by T d or nnz(Y)
        reg_z = reg_z_l2 * (1 - eta_Z) + reg_z_ar * eta_Z
        objective += reg_z * coef
    # end if

    return 0.5 * objective


# In[48]:
def z_step_tron_grad(z, Y, F, phi, C_Z, eta_Z):
    """Compute the gradient of the z-step objective."""
    n_samples, n_targets = Y.shape
    n_components, n_order = phi.shape

    Z = z.reshape(n_samples, n_components)
    if sp.issparse(Y):
        coef = C_Z * Y.nnz / (n_samples * n_components)

        grad = safe_sparse_dot(csr_gemm(1, Z, F, -1, Y.copy()), F.T)
        grad += (1 - eta_Z) * coef * Z

    else:
        coef = C_Z * n_targets / n_components

        YFT, FFT = np.dot(Y, F.T), np.dot(F, F.T)
        if C_Z > 0 and eta_Z < 1:
            FFT.flat[::n_components + 1] += (1 - eta_Z) * coef

        grad = np.dot(Z, FFT) - YFT
    # end if

    if C_Z > 0 and eta_Z > 0:
        ratio = n_samples / (n_samples - n_order)
        grad += ar_grad(Z, phi) * (ratio * eta_Z * coef)

    return grad.reshape(-1)


# In[49]:
def z_step_tron_hess(v, Y, F, phi, C_Z, eta_Z):
    """Compute the Hessian-vector product of the z-step objective for v."""
    n_samples, n_targets = Y.shape
    n_components, n_order = phi.shape

    V = v.reshape(n_samples, n_components)
    if sp.issparse(Y):
        coef = C_Z * Y.nnz / (n_samples * n_components)

        hess_v = safe_sparse_dot(csr_gemm(1, V, F, 0, Y.copy()), F.T)
        hess_v += (1 - eta_Z) * coef * V

    else:
        coef = C_Z * n_targets / n_components

        FFT = np.dot(F, F.T)
        if C_Z > 0 and eta_Z < 1:
            FFT.flat[::n_components + 1] += (1 - eta_Z) * coef

        hess_v = np.dot(V, FFT)
    # end if

    if C_Z > 0 and eta_Z > 0:
        # should call ar_hess_vect(V, Z, adj) but no Z is available
        ratio = n_samples / (n_samples - n_order)
        hess_v += ar_grad(V, phi) * ratio * eta_Z * coef

    return hess_v.reshape(-1)


# In[50]:
def z_step_tron(Z, Y, F, phi, C_Z, eta_Z, rtol=5e-2, atol=1e-4, verbose=False):
    """TRON solver for the f-step minimization problem."""
    f_call = z_step_tron_valh, z_step_tron_grad, z_step_tron_hess

    tron(f_call, Z.ravel(), n_iterations=5, rtol=rtol, atol=atol,
         args=(Y, F, phi, C_Z, eta_Z), verbose=verbose)

    return Z


def z_step_ncg_hess_(Z, v, Y, F, phi, C_Z, eta_Z):
    """A wrapper of the hess-vector product for ncg calls."""
    return z_step_tron_hess(v, Y, F, phi, C_Z, eta_Z)


def z_step_ncg(Z, Y, F, phi, C_Z, eta_Z, **kwargs):
    """Solve the Z-step using scipy's Newton-CG."""
    ZZ = fmin_ncg(f=z_step_tron_valh, x0=Z.ravel(), disp=False,
                  fprime=z_step_tron_grad, fhess_p=z_step_ncg_hess_,
                  args=(Y, F, phi, C_Z, eta_Z))
    return ZZ.reshape(Z.shape)


def z_step_lbfgs(Z, Y, F, phi, C_Z, eta_Z, **kwargs):
    """Solve the Z-step using scipy's L-BFGS method."""
    ZZ, f, d = fmin_l_bfgs_b(func=z_step_tron_valh, x0=Z.ravel(), iprint=0,
                             fprime=z_step_tron_grad,
                             args=(Y, F, phi, C_Z, eta_Z))

    return ZZ.reshape(Z.shape)


def z_step(Z, Y, F, phi, C_Z, eta_Z, kind="tron", **kwargs):
    """A common subroutine solving the Z-step minimization problem."""
    if kind == "tron":
        Z = z_step_tron(Z, Y, F, phi, C_Z, eta_Z, **kwargs)
    elif kind == "ncg":
        Z = z_step_ncg(Z, Y, F, phi, C_Z, eta_Z, **kwargs)
    elif kind == "lbfgs":
        Z = z_step_lbfgs(Z, Y, F, phi, C_Z, eta_Z, **kwargs)
    else:
        raise ValueError(f"""Unrecognized method `{kind}`""")

    return Z
