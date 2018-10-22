import timeit
import numpy as np
from scipy import sparse
from Infrastructure.numeric_schemes import AdvectionEqForwardEuler
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def direct_powers(n):
  a = np.exp(2 * np.pi * 1j / n)
  p = np.power(a, list(range(1, n)))
  p = [1] + p.tolist()
  return p


def direct_prod(n, v):
    a = 2
    diagonals = [np.ones(n), a * np.ones(n - 1), -a * np.ones(n - 1), [-a], [a]]
    A = sparse.diags(diagonals, offsets=[0, 1, -1, n - 1, -n + 1]).toarray()
    result = A.dot(v)
    return result, A


def deco_product(n, v):
    a = 2
    unity_roots = np.array(direct_powers(n)[1:])
    eigen = a * (unity_roots - np.flip(unity_roots, axis=0)) + 1
    eigen = np.conj(eigen)
    eigen = np.append([1], eigen)
    result = np.fft.fft(np.multiply(eigen, np.fft.ifft(v)))
    return result, eigen


def conv_product(n, v):
    a = 2
    v_list = v.tolist()
    res = []
    i1 = 0
    i2 = 1
    i3 = n - 1
    k = 0
    for vk in v_list:
        resk = 1 * v[i1] + a * v[i2] - a * v[i3]
        i1 += 1
        i2 += 1
        i3 += 1
        if k == 0:
            i3 = 0
        elif k == n - 2:
            i2 = 0
        k += 1
        res.append(resk)
    res = np.array(res)
    return res


def main():
    start_condition = lambda x: np.cos(2 * np.pi * x)
    gamma = 0.1
    N = 16
    dx = 1 / (N + 1)
    dt = gamma * dx
    last_t = 5
    x = np.arange(0, 1, dx)
    t = np.arange(0, last_t, dt)
    X, T = np.meshgrid(x, t)
    current_state = start_condition(x.reshape((1, -1)))
    numeric_solution = deepcopy(current_state)

    model = AdvectionEqForwardEuler(N, dt, last_t, first_x=0, last_x=1, starting_condition_func=start_condition)
    for _ in t.tolist()[1:]:
        current_state = model.make_step().reshape((1, -1))
        numeric_solution = np.append(numeric_solution, current_state, axis=0)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, T, numeric_solution, rstride=1, cstride=1)
    plt.show()
    print('t')


def test():
    n = 2 ** 4
    v = np.random.rand(n)
    glob = {'v': v, 'n': n}
    repeats = 1000
    t1 = timeit.timeit("deco_product(n, v)", setup="from __main__ import deco_product", globals=glob, number=repeats) / repeats
    t2 = timeit.timeit("direct_prod(n, v)", setup="from __main__ import direct_prod", globals=glob, number=repeats) / repeats
    t3 = timeit.timeit("conv_product(n, v)", setup="from __main__ import conv_product", globals=glob, number=repeats) / repeats
    print(t1 * 10 ** 6)
    print(t2 * 10 ** 6)
    print(t3 * 10 ** 6)


if __name__ == '__main__':
    main()
