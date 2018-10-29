import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from memory_profiler import profile

from Infrastructure.numeric_schemes import AdvectionEqForwardEuler, AdvectionModelLeapFrog
from Infrastructure.circulant_sparse_matrix import CirculantSparseMatrix


def exact_solution(x, t):
    return np.cos(2 * np.pi * (x + t))


def start_conditions(x):
    return exact_solution(x, 0)


def plot_result(data, x_grid, t_grid, gamma, N):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ylabel('T')
    plt.title("Approximation of {0} points, gamma={1}".format(N, gamma))
    plt.xlabel('X')
    ax.plot_surface(x_grid, t_grid, data, rstride=1, cstride=1)
    plt.show()


def perform_experiment(N, dt, last_t, first_x, last_x, nonhomogeneous_term):

    model = AdvectionModelLeapFrog(N, dt, last_t, first_x=first_x, last_x=last_x,
                                   starting_condition_func=start_conditions, nonhomogeneous_term=nonhomogeneous_term)
    times = int(np.floor(last_t / dt))
    for _ in range(times - 1):
        model.make_step()

    return model.current_state


def main():
    gammas = [0.1]
    n = np.power(2, list(range(8, 9)))

    first_x = 0
    last_x = 1
    last_t = 5

    nonhomogeneous_term = lambda y, t: np.zeros(y.shape)

    for gamma in gammas:
        approximation_errors = []
        deltaxs = []

        for N in n:
            dx = (last_x - first_x) / (N + 1)
            dt = gamma * dx ** 2
            x = np.arange(first_x, last_x, dx)
            last_calculated_time = np.floor(last_t / dt) * dt

            numeric_approximation = perform_experiment(N, dt, last_t, first_x, last_x, nonhomogeneous_term)
            exact_values = exact_solution(x, last_calculated_time)
            #plt.plot(x, numeric_approximation, label='Numeric Solution')
            #plt.plot(x, exact_values, label='Exact Solution')
            #plt.legend()
            #plt.show()
            error = np.linalg.norm(exact_values - numeric_approximation) * np.sqrt(dx)
            approximation_errors.append(error)
            deltaxs.append(dx)

        #plt.plot(np.log10(np.flip(deltaxs, axis=0)), np.log10(np.flip(approximation_errors, axis=0)))
        print(np.polyfit(np.log10(np.flip(deltaxs, axis=0)), np.log10(np.flip(approximation_errors, axis=0)), deg=1))
        #plt.show()


if __name__ == '__main__':
    main()
