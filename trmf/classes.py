import numpy as np

import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.utils import check_consistent_length, check_array

from .base import trmf, trmf_forecast_factors
from .base import trmf_forecast_targets


class TRMFRegressor(BaseEstimator):
    r"""Time-Series Regularized Matrix Factorization with regression.

    Finds two matrices (Z, F) with F being nonnegative and Z behaving
    like an autoregressive process, whose product approximates the target
    matrix Y, see [1]_.

    This implementation supports optional exogenous regressors, that can
    be used for predicting the target matrix (if fit_regression is True).
    This factorization can be used, for example, for dimensionality reduction
    in multivariate time-series, when it is important to be able to forecast
    future dynamics. The objective function is for the decomposition with
    (optional) intercept \mu, exogenous regressors X and their coefficients
    B is ::

    .. math::

        \frac{1}{2 T n} \| Y - Z F - X B + \mu\|_{fro}^2
        + \frac{C_{\phi}}{2 d p} \| \phi \|_{fro}^2
        + \frac{C_B}{2 m n} \| B \|_{fro}^2
        + \frac{C_Z}{2} (
            \frac{1 - \eta_Z}{T d} \|Z\|_{fro}^2
            + \frac{\eta_Z}{(T - p) d} AR_p(Z))
        + \frac{C_F}{2} * (
            \frac{1 - \eta_F}{d n} \|F\|_{fro}^2
            + \frac{\eta_F}{d n} R_G(F))

    Where::
        :math:`\|A\|_{fro}^2 = \sum_{i,j} A_{ij}^2` (Frobenius norm)

        :math:`AR_p(Z) = \sum_{j,t} (Z_{tj} - \hat{Z}_{t,j\mid t-1})^2`
        (Autoregressive regularizer of order p)

        :math:`\hat{Z}_{t,j\mid t-1} = \sum_{k} \phi_{jk} * Z_{t-k,j}`
        (One-step ahead autoregressive forecast)

        :math:`R_G(F) = \sum_{j\in G} \| F_{.j} - \bar{F}_{.j} \|^2`
        (L2 graph adjacency regularizer)

        :math:`\bar{F}_{.j} = \frac{1}{|G_j|} \sum_{k \in G_j} W_{jk} F_{.k}`
        (the weighted average over the endpoints of the outgoing edges
            from j)

    The objective function is minimized by cycling over minimization steps
    with respect to \mu, B, factorization (Z, F) and \phi.

    Parameters
    ----------
    n_components : int
        The number of latent autoregressive factors Z in the decompositon.

    n_order : int
        The assumed order of the latent autoregressive factors Z in the
        decompositon.

    C_Z : double, optional (default=0.1)
        Penalty parameter C_Z of the latent factor Z regularizer.

    C_F : double, optional (default=0.1)
        Penalty parameter C_F of the factor loadings F regularizer.

    C_phi : double, optional (default=0.01)
        Penalty parameter C_phi of the Ridge regilarizer for the
        coefficients of the autoregressive dynamics of the latent
        factors.

    eta_Z : double, optional (default=0.9)
        The regularization mixing parameter, with 0 <= eta_Z <= 1.
        For eta_Z = 0 the penalty on the factors Z is an elementwise
        L2 penalty.
        For eta_Z = 1 it is the L2 loss of an autoregressive forecasitng
        model of order `n_order` on the latent factors Z.
        For 0 < eta_Z < 1, the penalty is a combination of both.

    eta_F : double, optional (default=0.0)
        The regularization mixing parameter, with 0 <= eta_F <= 1.
        For eta_F = 0 the penalty on the factor loadings F is an
        elementwise L2 penalty.
        For eta_F = 1 it is the L2 graph adjacency regularizer.
        For 0 < eta_F < 1, the penalty is a combination of both.

    adj : sparse_matrix, optional (default=None)
        The precomputed adjacency matrix of a binary relation between
        the time series (columns of the matrix to be decomposed). Must
        be sparse and have at least as many rows and columns as there
        is time-series.

    C_B : double, optional (default=1.0)
        Penalty parameter C_B of the error term.

    fit_regression : boolean, optional (default=False)
        Whether to perform matrix decomposition on the residuals of
        the linear regression of a mutlivariate time-series on the
        provided exogenous data. If set to false, the fit method
        expects only the time-series matrix in `X` argument. If set
        to true, then the fit method expects the exogenous regressors
        and the multivariate time-series are provided in `X` and `y`,
        respectively.

    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations (i.e.
        data is expected to be already centered).

    nonnegative_factors : bool, (default=True)
        Whether to impose the non-negativity constraint on the estimated
        loadings of the latent autoregressive factros. If set to true,
        uses Fast prox method for
        then leading to a much faster algorithm.

    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria.

    n_max_iterations : int, (default=1000)
        The maximum number of iterations to be run.

    n_max_mf_iter : int, (default=5)
        The maximum number of the inner matrix decomposition iterations
        to be run.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data for the dual coordinate descent (if ``dual=True``). When
        ``dual=False`` the underlying implementation of :class:`LinearSVC`
        is not random and ``random_state`` has no effect on the results.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    factors_ : array, shape = (n_samples, n_components)
        The estimates of the time-series of the latent factors.

    loadings_ : array, shape = (n_components, n_targets)
        The estimated factor loadings.

    ar_coef_ : array, shape = (n_components, n_order)
        The estimated coefficients of the latent autoregressive dynamics.
        The coefficients are stored in the reverse order of recency: the
        values in each row are ordered from lag `n_order` (the least recent)
        up to lag `1` (the most recent).

    coef_ : array, optional, shape = (n_features, n_targets)
        Optional coefficients of the regression on the exogenous data.

    intercept_ : array, optional, shape = (1, n_targets)
        Optional estimated constant term in decomposition.

    References
    ----------
    .. [1] Yu, H. F., Rao, N., & Dhillon, I. S., (2016). "Temporal
           regularized matrix factorization for high-dimensional time
           series prediction." In Advances in neural information processing
           systems (pp. 847-855).
    """
    def __init__(self,
                 n_components,
                 n_order,
                 C_Z=1e-1,
                 C_F=1e-1,
                 C_phi=1e-2,
                 eta_Z=0.5,
                 eta_F=0.,
                 adj=None,
                 C_B=0.0,
                 fit_regression=False,
                 fit_intercept=True,
                 nonnegative_factors=True,
                 tol=1e-5,
                 n_max_iterations=1000,
                 n_max_mf_iter=5,
                 random_state=None):
        super(TRMFRegressor, self).__init__()

        self.n_components = n_components
        self.n_order = n_order
        self.C_Z = C_Z
        self.C_F = C_F
        self.C_phi = C_phi
        self.eta_Z = eta_Z
        self.eta_F = eta_F
        self.adj = adj
        self.C_B = C_B
        self.fit_regression = fit_regression
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.n_max_iterations = n_max_iterations
        self.n_max_mf_iter = n_max_mf_iter
        self.nonnegative_factors = nonnegative_factors
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        """Fit the TRMF regression model to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
            Training multivariate time series data either exogenous
            regressors or the mutlivariate time-series themselves, depending
            on the `fit_regression` setting.
            For fit_regression=True, the expected shape of X is
            (n_samples, n_features).
            For fit_regression=False, the expected shape of X is
            (n_samples, n_targets).

        y : None or array-like, shape (n_samples, n_targets)
            If fit_regression=True, then y is the target mutlivariate
            time-series, where n_samples is the number of observations
            in the time-series and n_targets is the number of observed
            series.
            For fit_regression=False, y must be `None`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # `X` is a matrix anyway (under any mode). no ensure_min_features=0
        #  since regression mode still requires at least one feature column.
        X = check_array(X, dtype="numeric", accept_sparse=False,
                        ensure_min_samples=self.n_order + 1, ensure_2d=True)
        if self.fit_regression:
            y = check_array(y, dtype="numeric", accept_sparse=False,
                            ensure_min_samples=self.n_order + 1,
                            ensure_2d=True)
        else:
            if y is not None:
                raise TypeError("""Exogenous regressors provided in `X`, """
                                """yet `fit_regression` is false.""")

            X, y = np.empty((X.shape[0], 0)), X
        # end if

        check_consistent_length(X, y)

        f_step_kind = "fgm" if self.nonnegative_factors else "tron"
        estimates = trmf(y, self.n_components, self.n_order, self.C_Z,
                         self.C_F, self.C_phi, self.eta_Z, self.eta_F,
                         adj=self.adj, fit_intercept=self.fit_intercept,
                         regressors=X, C_B=self.C_B, tol=self.tol,
                         n_max_iterations=self.n_max_iterations,
                         n_max_mf_iter=self.n_max_mf_iter,
                         f_step_kind=f_step_kind,
                         random_state=self.random_state)

        # Record the estimates in this instance's properties
        factors, loadings, ar_coef, intercept, beta = estimates

        self.factors_, self.loadings_ = factors, loadings
        self.ar_coef_ = ar_coef

        self.coef_, self.intercept_ = beta, intercept

        # self.fitted_ = np.dot(X, beta) + np.dot(factors, loadings) \
        #                + intercept

        return self

    def forecast_factors(self, n_ahead):
        r"""Compute the dynamic forecast of the latent factor time-series.

        Parameters
        ----------
        n_ahead : int
            The depth of the latent factors' forecast into the future.

        Details
        -------
        This computes a dynamic forecast of AR(p) process with :math:`p`
        equal to `n_order` and coefficients :math:`\phi` fixed at `ar_coef_`::

        .. math ::

            \hat{Z}_{t+h, j \mid t}
                = \sum_{k=1}^p \phi_{jk} * Z_{t + h - k,j \mid t}

        where::
            :math:`\hat{Z}_{s,j \mid t}` is :math:`Z_{s,j}` whenever
            :math:`t \leq s`.

        """

        return trmf_forecast_factors(n_ahead, self.ar_coef_,
                                     prehist=self.factors_)

    def predict(self, X=None, n_ahead=10):
        r"""Predict the targets based on the autoregressive decomposition.

        Parameters
        ----------
        X : None or array-like, shape (n_ahead, n_features)
            The future dynamics of the exogenous regressors or `None`,
            depending on the `fit_regression` setting.
            For fit_regression=True, the expected shape of X is
            (n_ahead, n_features).
            For fit_regression=False, X is expected to be `None`.

        n_ahead : int, optional (default=10)
            The depth of the latent factors' forecast into the future.

        Details
        -------
        This computes the prediction of the values of the time-series
        based on the `n`_ahead`-step dynamic forecasts of the latent
        factors and the regression coefficients.
        """

        if self.fit_regression:
            X = check_array(X, dtype="numeric", accept_sparse=False)
        else:
            X = np.empty((n_ahead, 0))
        # end if

        return trmf_forecast_targets(
            n_ahead, self.loadings_, self.ar_coef_, self.intercept_,
            self.coef_, self.factors_, regressors=X, mode="exog")
