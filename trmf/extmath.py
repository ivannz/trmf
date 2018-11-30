# updated version of `trmf_v0.8.2-sparse.ipynb` as of 2018-11-26 12:00

import numpy as np
import numba as nb

from sklearn.utils.extmath import safe_sparse_dot


# In[26] -- sparse:
@nb.njit("(float64, float64[:,::1], float64[:,::1], "
         "float64, int32[::1], int32[::1], float64[::1])",
         fastmath=True, error_model="numpy", parallel=False, cache=False)
def _csr_gemm(alpha, X, D, beta, Sp, Sj, Sx):
    # computes\mathcal{P}_\Omega(X D) -- n1 x n2 sparse matrix
    if abs(beta) > 0:
        for i in nb.prange(len(X)):
            # compute e_i' XD e_{Sj[j]}
            for j in range(Sp[i], Sp[i+1]):
                dot = np.dot(X[i], D[:, Sj[j]])
                Sx[j] = beta * Sx[j] + alpha * dot
        # end for
    else:
        for i in nb.prange(len(X)):
            # compute e_i' XD e_{Sj[j]}
            for j in range(Sp[i], Sp[i+1]):
                Sx[j] = alpha * np.dot(X[i], D[:, Sj[j]])
        # end for
    # end if


def csr_gemm(alpha, X, D, beta, Y):
    _csr_gemm(alpha, X, D, beta, Y.indptr, Y.indices, Y.data)
#     dot = alpha * np.dot(X, D)[Y.nonzero()]
#     if abs(beta) > 0:
#         Y.data = beta * Y.data + dot
#     else:
#         Y.data = dot
    return Y


def csr_column_means(X):
    f_sums = np.bincount(X.indices, weights=X.data, minlength=X.shape[1])
    n_nnz = np.maximum(np.bincount(X.indices, minlength=X.shape[1]), 1.)

    return (f_sums / n_nnz)[np.newaxis]
