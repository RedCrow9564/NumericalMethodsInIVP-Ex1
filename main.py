import timeit
import numpy as np
from scipy import sparse


def main():
    gamma = 1
    N = 5
    last_t = 5

    dx = 1 / (N + 1)
    dt = gamma * dx

    a = dt/(2 * dx)
    main_diag = np.ones(N + 1)
    upper_main_diag = a * np.ones(N + 1)
    under_main_diag = -a * np.ones(N + 1)
    top_element = [-a]
    bottom_element = [a]
    diagonals = [main_diag, upper_main_diag, under_main_diag, top_element, bottom_element]
    offsets = np.array([0, 1, -1, N, -N])
    b = sparse.diags(diagonals, offsets, shape=(N + 1, N + 1), format='dia').tocsr()

    v = np.ones(N + 1)
    return b, v


def sparse_product(b, v):
    b ** 2


if __name__ == '__main__':
    b, v = main()
    print(b.toarray())
    iterations = 2000
    times = timeit.repeat('sparse_product(b, v)', 'from __main__ import main, sparse_product', repeat=3,
                          number=iterations, globals={'b': b, 'v': v})
    print('Program took {0} usec'.format(min(times) * 1000000 / iterations))

    b = b.toarray()
    times = timeit.repeat('sparse_product(b, v)', 'from __main__ import main, sparse_product', repeat=3,
                          number=iterations, globals={'b': b, 'v': v})
    print('Program took {0} usec'.format(min(times) * 1000000 / iterations))
