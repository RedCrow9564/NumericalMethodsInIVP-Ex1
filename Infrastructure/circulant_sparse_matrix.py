import numpy as np
from numpy.fft import fft, ifft
from Infrastructure.circulant_sparse_product import compute
#from memory_profiler import profile


class CirculantSparseMatrix(object):

    def __init__(self, n, nonzero_terms, nonzero_indices):
        self._n = n
        self._terms = np.array(nonzero_terms)
        self._indices = np.array(nonzero_indices)
        self._nonzero_len = len(nonzero_indices)
        row = np.array(n * [0], dtype=np.float)
        row[nonzero_indices] = self._terms
        self._eigs = fft(row)

    def dot(self, v, steps_num):
        if steps_num > 1:
            next_state = np.real(fft(np.multiply(np.power(self._eigs, steps_num), ifft(v))))
        else:
            next_state = np.zeros(v.shape)
            compute(self._terms, self._indices, v, next_state)
        return next_state
