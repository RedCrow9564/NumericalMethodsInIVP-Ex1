import numpy as np
from copy import deepcopy
from memory_profiler import profile

from Infrastructure.circulant_sparse_matrix import CirculantSparseMatrix


class _ModelTemplate(object):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        x_samples = np.linspace(first_x, last_x, n + 1)
        t_samples = np.arange(0, last_t, dt)
        self._current_step = 0
        self._dx = x_samples[1] - x_samples[0]
        self._dt = dt
        self._n = n
        self.current_state = starting_condition_func(x_samples)
        x_grid, t_grid = np.meshgrid(x_samples, t_samples)
        self._sampled_nonhomogeneous_term = nonhomogeneous_term(x_grid, t_grid)
        del x_grid
        del t_grid
        del x_samples
        del t_samples

    def make_step(self):
        raise NotImplementedError("Each model MUST implement this method!")


class _ForwardEulerModel(_ModelTemplate):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func, nonhomogeneous_term, transition_mat):
        super(_ForwardEulerModel, self).__init__(n, dt, last_t, first_x, last_x, starting_condition_func,
                                                 nonhomogeneous_term)
        self._transition_mat = transition_mat

    def make_step(self):
        non_homogeneous_element = self._sampled_nonhomogeneous_term[self._current_step, :]
        self.current_state = self._transition_mat.dot(self.current_state) + self._dt * non_homogeneous_element
        self._current_step += 1


class _LeapFrogModel(_ModelTemplate):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func, nonhomogeneous_term, transition_mat):
        super(_LeapFrogModel, self).__init__(n, dt, last_t, first_x, last_x, starting_condition_func,
                                             nonhomogeneous_term)
        self._transition_mat = transition_mat
        self._first_step_model = AdvectionEqForwardEuler(n, dt, last_t, first_x, last_x, starting_condition_func)
        self._previous_state = deepcopy(self.current_state)

    def make_step(self):
        if self._current_step == 0:
            self._first_step_model.make_step()
            del self._first_step_model
        else:
            non_homogeneous_element = self._sampled_nonhomogeneous_term[self._current_step, :]
            current_state_copy = deepcopy(self.current_state)
            self.current_state = self._previous_state + self._transition_mat.dot(self.current_state) + \
                self._dt * non_homogeneous_element
            self._previous_state = current_state_copy

        self._current_step += 1


class AdvectionEqForwardEuler(_ForwardEulerModel):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func):
        nonhomogeneous_term = lambda x, t: np.zeros(x.shape)  # The equation is assumed to be homogeneous.
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / (2 * dx)
        transition_mat = CirculantSparseMatrix(n + 1, [1, ratio, -ratio], [0, 1, n])
        super(AdvectionEqForwardEuler, self).__init__(n, dt, last_t, first_x, last_x, starting_condition_func,
                                                      nonhomogeneous_term, transition_mat)


class AdvectionModelLeapFrog(_LeapFrogModel):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func):
        nonhomogeneous_term = lambda x, t: np.zeros(x.shape)  # The equation is assumed to be homogeneous.
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / dx
        transition_mat = CirculantSparseMatrix(n + 1, [ratio, -ratio], [1, n])
        super(AdvectionModelLeapFrog, self).__init__(n, dt, last_t, first_x, last_x, starting_condition_func,
                                                     nonhomogeneous_term, transition_mat)