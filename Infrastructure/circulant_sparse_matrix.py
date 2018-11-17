import numpy as np
from Infrastructure.circulant_sparse_product import compute


class CirculantSparseMatrix(object):

    def __init__(self, n, nonzero_terms, nonzero_indices):
        self._n = n
        self._terms = np.array(nonzero_terms)
        self._indices = np.array(nonzero_indices)

    def dot(self, v):
        next_state = np.zeros(v.shape)
        compute(self._terms, self._indices, v, next_state)
        return next_state
