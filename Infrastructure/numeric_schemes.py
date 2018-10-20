import numpy as np
from numpy.fft import fft, ifft

class _ModelTemplate(object):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        self.x_samples = np.linspace(first_x, last_x, n)
        self.t_samples = np.arange(0, last_t, dt)
        self._dx = self.x_samples[1] - self.x_samples[0]
        self._dt = dt
        self._n = n
        self._current_state = starting_condition_func(self.x_samples)
        self._nonhomogeneous_term = nonhomogeneous_term

    def make_steps(self, steps_num):
        raise NotImplementedError("Each model MUST implement this method!")


class AdvectionEqForwardEuler(_ModelTemplate):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func):
        nonhomogeneous_term = 0  # The equation is assumed to be homogeneous.
        super(AdvectionEqForwardEuler, self).__init__(n, dt, last_t, first_x, last_x, starting_condition_func,
                                                      nonhomogeneous_term)

        ratio = self._dt / (2 * self._dx)
        unity_roots = roots_of_unity(n)[1:]
        self._transition_mat_eigenvalues = 1 + ratio * (unity_roots - np.flip(unity_roots, axis=0))

    def make_steps(self, steps_num=1):
        eigenvalues_power = np.power(self._transition_mat_eigenvalues, np.mod(steps_num, self._n))
        self._current_state = fft(np.multiply(eigenvalues_power, ifft(self._current_state)))
        return self._current_state


def roots_of_unity(n):
    primitive_root = np.exp(2 * np.pi * 1j / n)
    roots = np.power(primitive_root, list(range(1, n)))
    roots = np.append([1], roots)
    return roots
