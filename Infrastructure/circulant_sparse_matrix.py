import numpy as np
from scipy.sparse import diags
from Infrastructure.circulant_sparse_product import compute, solve_almost_tridiagonal_system


class CirculantSparseMatrix(object):

    def __init__(self, n, nonzero_terms, nonzero_indices):
        self._n = n
        self._terms = np.array(nonzero_terms)
        self._indices = np.array(nonzero_indices, dtype=np.int32)

    def dot(self, v):
        next_state = np.zeros(v.shape)
        compute(self._terms, self._indices, v, next_state)
        return next_state


class AlmostTridiagonalToeplitzMatrix(CirculantSparseMatrix):
    def __init__(self, n, nonzero_terms):
        super(AlmostTridiagonalToeplitzMatrix, self).__init__(n, nonzero_terms, nonzero_indices=[0, 1, n])
        diagonals = [nonzero_terms[0] * np.ones((1, n), dtype=np.float32)[0],
                     nonzero_terms[1] * np.ones((1, n - 1), dtype=np.float32)[0],
                     nonzero_terms[2] * np.ones((1, n - 1), dtype=np.float32)[0],
                     [nonzero_terms[1]], [nonzero_terms[2]]]
        self._diag_term = nonzero_terms[0]
        self._up_diag = nonzero_terms[1]
        self._sub_diag = nonzero_terms[2]
        self._mat = diags(diagonals, [0, 1, -1, -n + 1, n - 1]).toarray()
        self._inverse = np.linalg.inv(self._mat)

    def inverse_solution(self, v):
        solve_almost_tridiagonal_system(self._diag_term, self._sub_diag, self._up_diag, self._n, v,
                                        np.zeros(v.shape))
        return v


