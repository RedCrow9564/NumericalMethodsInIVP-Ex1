cimport cython

ctypedef fused possible_type:
    int
    double
    long
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#@cython.language_level('3')
def compute(possible_type[::1] circulant_terms, int[::1] terms_indices, possible_type[::1] current_state,
            possible_type[::1] next_state):
    cdef Py_ssize_t terms_num = circulant_terms.shape[0]
    cdef Py_ssize_t index_to_reset = terms_num - 1
    cdef int[::1] current_indices = terms_indices
    cdef Py_ssize_t i, k, n = current_state.shape[0]

    for k in range(n):
        for i in range(terms_num):
            next_state[k] += current_state[current_indices[i]] * circulant_terms[i]
            current_indices[i] += 1

        if current_indices[index_to_reset] == n:
            current_indices[index_to_reset] = 0
            index_to_reset -= 1
