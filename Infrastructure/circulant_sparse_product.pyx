# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, free

ctypedef fused possible_type:
	int
	double
	long
	long long

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def compute(possible_type[::1] circulant_terms, int[::1] terms_indices, possible_type[::1] current_state,
			possible_type[::1] next_state):
	cdef Py_ssize_t terms_num = circulant_terms.shape[0]
	cdef Py_ssize_t index_to_reset = terms_num - 1
	cdef int[::1] current_indices = terms_indices
	cdef Py_ssize_t i, k, n = current_state.shape[0]

	for k in range(n):
		next_state[k] = 0

		for i in range(terms_num):
			next_state[k] += current_state[current_indices[i]] * circulant_terms[i]
			current_indices[i] += 1

		if current_indices[index_to_reset] == n:
			current_indices[index_to_reset] = 0
			index_to_reset -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def solve_almost_tridiagonal_system(const double diag_element, const double sub_diag, const double up_diag,
									const int size, double[::1] current_state, double[::1] u):
	cdef double denom = 1
	cdef double factor = 0
	cdef double first_diag = 2 * diag_element
	cdef double last_diag = diag_element + sub_diag * up_diag / diag_element
	cdef Py_ssize_t i = 0

	u[0] = -diag_element
	u[size - 1] = up_diag
	for i in range(1, size - 1):
		u[i] = 0

	ThomasAlgorithm(sub_diag, diag_element, up_diag, current_state, size, first_diag, last_diag)  # inv(B) * y
	ThomasAlgorithm(sub_diag, diag_element, up_diag, u, size, first_diag, last_diag)  # inv(B) * u
	denom = 1 + u[0] - sub_diag * u[size - 1] / diag_element
	factor = (current_state[0] - sub_diag * current_state[size - 1] / diag_element) / denom


	for i in range(size):
		current_state[i] = current_state[i] - factor * u[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void ThomasAlgorithm(const double sub_diag_element, const double diag_element,
						  const double super_diag_element, double[::1] d, const int size,
						  const double first_diag_element, const double last_diag_element):
	cdef Py_ssize_t i
	cdef double* bprimes

	bprimes = <double*> malloc(sizeof(double) * size)
	bprimes[0] = first_diag_element
	for i in range(1, size - 1):
		bprimes[i] = diag_element - sub_diag_element * super_diag_element / bprimes[i - 1]
		d[i] -= sub_diag_element * d[i - 1] / bprimes[i - 1]

	bprimes[size - 1] = last_diag_element - sub_diag_element * super_diag_element / bprimes[size - 2]
	d[size - 1] -= sub_diag_element * d[size - 2] / bprimes[size - 2]

	d[size - 1] /= bprimes[size - 1]
	for i in range(size - 2, -1, -1):
		d[i] -= super_diag_element * d[i + 1]
		d[i] /= bprimes[i]

	free(bprimes)
