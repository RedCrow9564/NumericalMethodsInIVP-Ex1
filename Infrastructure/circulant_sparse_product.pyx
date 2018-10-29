cimport cython

ctypedef fused possible_type:
    int
    double
    long
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def compute(possible_type[::1] circulant_terms, int[::1] terms_indices, possible_type[::1] current_state,
            possible_type[::1] next_state, Py_ssize_t terms_num):
    cdef Py_ssize_t index_to_reset = terms_num - 1
    cdef int[::1] current_indices = terms_indices
    cdef Py_ssize_t i, k, n = current_state.shape[0]

    for k in range(n):
        next_state[k] = 0

        for i in range(terms_num):
            next_state[k] += current_state[current_indices[i]] * circulant_terms[i]

        if current_indices[index_to_reset] == n - 1:
            current_indices[index_to_reset] = -1
            index_to_reset -= 1

        for i in range(terms_num):
            current_indices[i] += 1
