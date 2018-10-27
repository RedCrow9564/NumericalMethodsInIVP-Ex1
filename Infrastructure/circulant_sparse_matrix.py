from copy import deepcopy
import numpy as np
from numpy.fft import fft, ifft
from memory_profiler import profile


class CirculantSparseMatrix(object):

    def __init__(self, n, nonzero_terms, nonzero_indices):
        self._n = n
        self._terms = np.array(nonzero_terms)
        self._indices = np.array(nonzero_indices)
        self._nonzero_len = len(nonzero_indices)
        row = np.array(n * [0], dtype=np.float)
        row[nonzero_indices] = self._terms
        self._eigs = fft(row)
        self._flag = True

    def dot(self, v):
        if self._flag:
            return np.real(fft(np.multiply(self._eigs, ifft(v))))
        else:
            index_to_reset = self._nonzero_len - 1
            current_indices = deepcopy(self._indices)
            result = []
            for _ in range(self._n):
                resk = self._terms.dot(v[current_indices])
                result.append(resk)
                if current_indices[index_to_reset] == self._n - 1:
                    current_indices[index_to_reset] = -1
                    index_to_reset -= 1
                current_indices += 1
            return np.array(result)
