# stripped version of `trmf_v0.8.2.ipynb` as of 2018-08-29 16:00

import numpy as np

import scipy.sparse as sp

from sklearn.utils import check_consistent_length, check_array

from .steps import f_step, z_step_tron, phi_step
from .steps import precompute_graph_reg
from .ext import b_step

from .extmath import csr_column_means, csr_gemm


# In[48]:
from sklearn.utils import check_random_state


def trmf_init(data, n_components, n_order, random_state=None):
    random_state = check_random_state(random_state)

    n_samples, n_targets = data.shape
    if sp.issparse(data):
        U, s, Vh = sp.linalg.svds(data, k=n_components)

        order = np.argsort(s)[::-1]
        U, s, Vh = U[:, order], s[order], Vh[order]
    else:
        U, s, Vh = np.linalg.svd(data, full_matrices=False)

    factors = U[:, :n_components].copy()
    loadings = Vh[:n_components].copy()
    loadings *= s[:n_components, np.newaxis]

    n_svd_factors = factors.shape[1]
    if n_svd_factors < n_components:
        random_factors = random_state.normal(
            scale=0.01, size=(n_samples, n_components - n_svd_factors))
        factors = np.concatenate([factors, random_factors], axis=1)

    n_svd_loadings = loadings.shape[0]
    if n_svd_loadings < n_components:
        random_loadings = random_state.normal(
            scale=0.01, size=(n_components - n_svd_loadings, n_targets))
        loadings = np.concatenate([loadings, random_loadings], axis=0)

    phi = np.zeros((n_components, n_order))
    ar_coef = phi_step(phi, factors, 1.0, 0., 1.0)
    return factors, loadings, ar_coef


# Used to be `In[49]:` but has been modified on the 28th of August
def trmf(data, n_components, n_order, C_Z, C_F, C_phi, eta_Z,
         eta_F=0., adj=None, fit_intercept=False, regressors=None, C_B=0.0,
         tol=1e-6, n_max_iterations=2500, n_max_mf_iter=5, f_step_kind="fgm",
         random_state=None):
    if not all(C >= 0 for C in (C_Z, C_F, C_phi, C_B)):
        raise ValueError("""Negative ridge regularizer coefficient.""")

    if not all(0 <= eta <= 1 for eta in (eta_Z, eta_F)):
        raise ValueError("""Share `eta` is not within `[0, 1]`.""")

    if not (n_components > 0):
        raise ValueError("""Empty latent factors are not supported.""")

    if C_F > 0 and eta_F > 0:
        if not sp.issparse(adj):
            raise TypeError("""The adjacency matrix must be sparse.""")

        # precompute the outbound average dsicrepancy operator
        adj = precompute_graph_reg(adj)
    # end if

    # validate the input data
    data = check_array(data, dtype="numeric", accept_sparse=True,
                       ensure_min_samples=n_order + 1)

    # prepare the regressors
    n_samples, n_targets = data.shape
    if isinstance(regressors, str):
        if regressors != "auto" or True:
            raise ValueError(f"""Invalid regressor setting `{regressors}`""")

        if sp.issparse(data):
            raise ValueError("""`data` cannot be sparse in """
                             """autoregression mode""")

        # assumes order-1 explicit autoregression
        regressors, data = data[:-1], data[1:]
        n_samples, n_targets = data.shape

    elif regressors is None:
        regressors = np.empty((n_samples, 0))

    # end if

    regressors = check_array(regressors, dtype="numeric",
                             accept_sparse=False, ensure_min_features=0)

    check_consistent_length(regressors, data)

    # by default the intercept is zero
    intercept = np.zeros((1, n_targets))

    # initialize the regression coefficients
    n_regressors = regressors.shape[1]
    beta = np.zeros((n_regressors, n_targets))

    # prepare smart guesses
    factors, loadings, ar_coef = trmf_init(data, n_components, n_order,
                                           random_state=random_state)

    # prepare for estimating the coefs of the exogenous ridge regression
    if fit_intercept and n_regressors > 0:
        regressors_mean = regressors.mean(axis=0, keepdims=True)
        regressors_cntrd = regressors - regressors_mean
    else:
        regressors_cntrd = regressors
    # end if

    # initialize the outer loop
    ZF, lip_f, lip_b = np.dot(factors, loadings), 500.0, 500.0
    ZF_old_norm, delta = np.linalg.norm(ZF, ord="fro"), +np.inf

    if sp.issparse(data):
        # run the trmf algo
        for iteration in range(n_max_iterations):
            if (delta <= ZF_old_norm * tol) and (iteration > 0):
                break

            # Fit the exogenous ridge-regression with an optional intercept
            if fit_intercept or n_regressors > 0:
                resid = csr_gemm(-1, factors, loadings, 1, data.copy())

                if fit_intercept:
                    intercept = csr_column_means(resid)

                if n_regressors > 0:
                    if fit_intercept:
                        resid.data -= intercept[0, resid.indices]
                    # end if

                    # solve for beta
                    beta, lip_b = b_step(beta, resid, regressors_cntrd, C_B,
                                         kind="tron")

                    # mean(R) - mean(X) beta = mu
                    if fit_intercept:
                        intercept -= np.dot(regressors_mean, beta)
                    # end if
                # end if

                # prepare the residuals for the trmf loop
                resid = data.copy()
                if n_regressors > 0:
                    resid = csr_gemm(-1, regressors, beta, 1, resid)

                if fit_intercept:
                    resid.data -= intercept[0, resid.indices]
            else:
                resid = data
            # end if

            # update (F, Z), then phi
            for inner_iter in range(n_max_mf_iter):
                loadings, lip_f = f_step(loadings, resid, factors, C_F, eta_F,
                                         adj, kind=f_step_kind, lip=lip_f)

                factors = z_step_tron(factors, resid, loadings, ar_coef,
                                      C_Z, eta_Z)
            # end for

            if n_order > 0:
                ar_coef = phi_step(ar_coef, factors, C_Z, C_phi, eta_Z)
            # end if

            # recompute the reconstruction and convergence criteria
            ZF, ZF_old = np.dot(factors, loadings), ZF
            delta = np.linalg.norm(ZF - ZF_old, ord="fro")
            ZF_old_norm = np.linalg.norm(ZF_old, ord="fro")
        # end for
    else:
        # run the trmf algo
        for iteration in range(n_max_iterations):
            if (delta <= ZF_old_norm * tol) and (iteration > 0):
                break

            # Fit the exogenous ridge-regression with an optional intercept
            if fit_intercept or n_regressors > 0:
                resid = data - ZF
                if fit_intercept:
                    intercept = resid.mean(axis=0, keepdims=True)
                # end if

                if n_regressors > 0:
                    if fit_intercept:
                        resid -= intercept
                    # end if

                    # solve for beta
                    beta, lip_b = b_step(beta, resid, regressors_cntrd, C_B,
                                         kind="tron")

                    # mean(R) - mean(X) beta = mu
                    if fit_intercept:
                        intercept -= np.dot(regressors_mean, beta)
                    # end if
                # end if

                resid = data.copy()
                if n_regressors > 0:
                    resid -= np.dot(regressors, beta)

                if fit_intercept:
                    resid -= intercept
            else:
                resid = data
            # end if

            # update (F, Z), then phi
            for inner_iter in range(n_max_mf_iter):
                loadings, lip_f = f_step(loadings, resid, factors, C_F, eta_F,
                                         adj, kind=f_step_kind, lip=lip_f)

                factors = z_step_tron(factors, resid, loadings, ar_coef,
                                      C_Z, eta_Z)
            # end for

            if n_order > 0:
                ar_coef = phi_step(ar_coef, factors, C_Z, C_phi, eta_Z)
            # end if

            # recompute the reconstruction and convergence criteria
            ZF, ZF_old = np.dot(factors, loadings), ZF
            delta = np.linalg.norm(ZF - ZF_old, ord="fro")
            ZF_old_norm = np.linalg.norm(ZF_old, ord="fro")
        # end for
    # end if

    return factors, loadings, ar_coef, intercept, beta


# Modified `In[50]`
def trmf_forecast_factors(n_ahead, ar_coef, prehist):
    n_components, n_order = ar_coef.shape
    if n_ahead < 1:
        raise ValueError("""`n_ahead` must be a positive integer.""")

    if len(prehist) < n_order:
        raise TypeError("""Factor history is too short.""")

    forecast = np.concatenate([
        prehist[-n_order:] if n_order > 0 else prehist[:0],
        np.zeros((n_ahead, n_components))
    ], axis=0)

    # compute the dynamic forecast
    for h in range(n_order, n_order + n_ahead):
        # ar_coef are stored in little endian lag order: from lag p to lag 1
        #  from the least recent to the most recent!
        forecast[h] = np.einsum("il,li->i", ar_coef, forecast[h - n_order:h])

    return forecast[-n_ahead:]


# Extra functionality
def trmf_forecast_targets(n_ahead, loadings, ar_coef, intercept, beta,
                          factors, regressors=None, mode="exog"):
    n_regressors, n_targets = beta.shape
    if regressors is None:
        if n_regressors > 0:
            raise TypeError("""Regressors must be provided.""")
        regressors = np.empty((n_ahead, 0))

    regressors = check_array(regressors, dtype="numeric",
                             accept_sparse=False, ensure_min_features=0)

    if regressors.shape[1] != n_regressors:
        raise TypeError("""Invalid number of regressor features.""")

    if mode == "exog":
        if regressors.shape[0] < n_ahead:
            raise TypeError("""Not enough future observations.""")

    elif mode == "auto":
        if n_regressors != n_targets:
            raise TypeError("""Invalid `beta` for mode `auto`.""")

        if regressors.shape[0] < 1:
            raise TypeError("""Insufficient history of targets.""")
    # end if

    # step 1: predict the latent factors
    forecast = trmf_forecast_factors(n_ahead, ar_coef, factors)
    factors_forecast = np.dot(forecast, loadings)

    # step 2: predict the targets
    if mode == "exog":
        # assume the regressors are exogenous
        targets = np.dot(regressors, beta) + factors_forecast + intercept

    elif mode == "auto":
        # Assume the regressors are order 1 autoregressors (can be
        #  order-q but needs embedding).
        targets = np.concatenate([
            regressors[-1:],
            np.zeros((n_ahead, n_regressors), dtype=regressors.dtype)
        ], axis=0)

        # compute the dynamic forecast
        for h in range(n_ahead):
            targets[h + 1] = intercept + np.dot(targets[h], beta) \
                             + factors_forecast[h]
        # end for
    # end if

    return targets[-n_ahead:]
