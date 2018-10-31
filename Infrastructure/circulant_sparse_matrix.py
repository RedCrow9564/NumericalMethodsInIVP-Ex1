import numpy as np
#from numpy.fft import fft, ifft
from Infrastructure.circulant_sparse_product import compute
#from memory_profiler import profile


class CirculantSparseMatrix(object):

    def __init__(self, n, nonzero_terms, nonzero_indices):
        self._n = n
        self._terms = np.array(nonzero_terms)
        self._indices = np.array(nonzero_indices)
        #row = np.array(n * [0], dtype=np.float)
        #row[nonzero_indices] = self._terms
        #self._eigs = fft(row)
        #self._flag = True

    def dot(self, v):
        #if self._flag:
        #    return np.real(fft(np.multiply(self._eigs, ifft(v))))
        #else:
        next_state = np.zeros(v.shape)
        compute(self._terms, self._indices, v, next_state)
        return next_state
