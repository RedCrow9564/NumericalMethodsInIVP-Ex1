import numpy as np
from copy import deepcopy
from enum import Enum
#from memory_profiler import profile

from Infrastructure.circulant_sparse_matrix import CirculantSparseMatrix, AlmostTridiagonalToeplitzMatrix


class _ModelTemplate(object):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        self._x_samples = np.linspace(first_x, last_x, n + 1, endpoint=False)
        self._nonhomogeneous_term = nonhomogeneous_term
        self._current_step = 0
        self._current_time = first_t
        self._dx = self._x_samples[1] - self._x_samples[0]
        self._dt = dt
        self._n = n
        self.current_state = starting_condition_func(self._x_samples)

    def make_step(self):
        raise NotImplementedError("Each model MUST implement this method!")


class _ForwardEulerModel(_ModelTemplate):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term, transition_mat):
        super(_ForwardEulerModel, self).__init__(n, dt, first_t, first_x, last_x, starting_condition_func,
                                                 nonhomogeneous_term)
        self._transition_mat = transition_mat

    def make_step(self):
        x_grid, current_t_grid = np.meshgrid(self._x_samples, self._current_time)
        non_homogeneous_element = self._nonhomogeneous_term(x_grid, current_t_grid)[0]
        self.current_state = self._transition_mat.dot(self.current_state) + self._dt * non_homogeneous_element
        self._current_step += 1
        self._current_time += self._dt


class _BackwardEulerModel(_ModelTemplate):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term, transition_mat):
        super(_BackwardEulerModel, self).__init__(n, dt, first_t, first_x, last_x, starting_condition_func,
                                                  nonhomogeneous_term)
        ratio = dt / self._dx
        self._transition_mat = AlmostTridiagonalToeplitzMatrix(n + 1, [None])


class _LeapFrogModel(_ModelTemplate):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term, transition_mat,
                 first_step_model):
        super(_LeapFrogModel, self).__init__(n, dt, first_t, first_x, last_x, starting_condition_func,
                                             nonhomogeneous_term)
        self._transition_mat = transition_mat
        self._first_step_model = first_step_model(n, dt, first_t, first_x, last_x, starting_condition_func,
                                                  nonhomogeneous_term)
        self._previous_state = deepcopy(self.current_state)

    def make_step(self):
        if self._current_step == 0:
            self._first_step_model.make_step()
            del self._first_step_model
        else:
            x_grid, current_t_grid = np.meshgrid(self._x_samples, self._current_time)
            non_homogeneous_element = self._nonhomogeneous_term(x_grid, current_t_grid)[0]
            current_state_copy = deepcopy(self.current_state)
            self.current_state = self._previous_state + self._transition_mat.dot(self.current_state) + \
                2 * self._dt * non_homogeneous_element
            self._previous_state = current_state_copy

        self._current_step += 1
        self._current_time += self._dt


class AdvectionModelForwardEuler(_ForwardEulerModel):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / (2 * dx)
        transition_mat = CirculantSparseMatrix(n + 1, [1, ratio, -ratio], [0, 1, n])
        super(AdvectionModelForwardEuler, self).__init__(n, dt, first_t, first_x, last_x,
                                                         starting_condition_func, nonhomogeneous_term, transition_mat)


class AdvectionModelLeapFrog(_LeapFrogModel):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / dx
        transition_mat = CirculantSparseMatrix(n + 1, [ratio, -ratio], [1, n])
        super(AdvectionModelLeapFrog, self).__init__(n, dt, first_t, first_x, last_x, starting_condition_func,
                                                     nonhomogeneous_term, transition_mat, AdvectionModelForwardEuler)


class AdvectionModelUpwindScheme(_ForwardEulerModel):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / dx
        transition_mat = CirculantSparseMatrix(n + 1, [-ratio + 1, ratio], [0, 1])
        super(AdvectionModelUpwindScheme, self).__init__(n, dt, first_t, first_x, last_x,
                                                         starting_condition_func, nonhomogeneous_term, transition_mat)


class AdvectionModelDownwindScheme(_ForwardEulerModel):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / dx
        transition_mat = CirculantSparseMatrix(n + 1, [ratio + 1, -ratio], [0, n])
        super(AdvectionModelDownwindScheme, self).__init__(n, dt, first_t, first_x, last_x,
                                                           starting_condition_func, nonhomogeneous_term, transition_mat)


class AdvectionModelLaxFriedrichs(_ForwardEulerModel):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / (2 * dx)
        transition_mat = CirculantSparseMatrix(n + 1, [ratio + 0.5, -ratio + 0.5], [1, n])
        super(AdvectionModelLaxFriedrichs, self).__init__(n, dt, first_t, first_x, last_x,
                                                          starting_condition_func, nonhomogeneous_term, transition_mat)


class AdvectionModelLaxWendroff(_ForwardEulerModel):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / dx
        coefficients = [1 - ratio ** 2, 0.5 * ratio + 0.5 * ratio ** 2, -0.5 * ratio + 0.5 * ratio ** 2]
        transition_mat = CirculantSparseMatrix(n + 1, coefficients, [0, 1, n])
        super(AdvectionModelLaxWendroff, self).__init__(n, dt, first_t, first_x, last_x,
                                                        starting_condition_func, nonhomogeneous_term, transition_mat)


class HeatModelForwardEuler(_ForwardEulerModel):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        dx = (last_x - first_x) / (n + 1)
        ratio = dt / (dx ** 2)
        transition_mat = CirculantSparseMatrix(n + 1, [-2 * ratio + 1, ratio, ratio], [0, 1, n])
        super(HeatModelForwardEuler, self).__init__(n, dt, first_t, first_x, last_x, starting_condition_func,
                                                     nonhomogeneous_term, transition_mat)


class HeatModelLeapFrog(_LeapFrogModel):
    def __init__(self, n, dt, first_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        dx = (last_x - first_x) / (n + 1)
        ratio = 2 * (dt / (dx ** 2))
        transition_mat = CirculantSparseMatrix(n + 1, [-2 * ratio, ratio, ratio], [0, 1, n])
        super(HeatModelLeapFrog, self).__init__(n, dt, first_t, first_x, last_x, starting_condition_func,
                                                     nonhomogeneous_term, transition_mat, HeatModelForwardEuler)


class ModelName(Enum):
    AdvectionEquation_ForwardEuler = "Advection Equation - Forward Euler"
    AdvectionEquation_LeapFrog = "Advection Equation - Leap Frog"
    AdvectionEquation_Upwind = "Advection Equation - Upwind Scheme"
    AdvectionEquation_Downwind = "Advection Equation - Downwind Scheme"
    AdvectionEquation_LaxFriedrichs = "Advection Equation - Lax Friedrichs"
    AdvectionEquation_LaxWendroff = "Advection Equation - Lax Wendroff"
    AdvectionEquation_BackwardEuler = "Advection Equation - Backward Euler"
    HeatEquation_ForwardEuler = "Heat Equation - Forward Euler"
    HeatEquation_LeapFrog = "Heat Equation - Leap Frog"


_models_names_to_objects = {
    ModelName.AdvectionEquation_ForwardEuler: AdvectionModelForwardEuler,
    ModelName.AdvectionEquation_LeapFrog: AdvectionModelLeapFrog,
    ModelName.AdvectionEquation_Upwind: AdvectionModelUpwindScheme,
    ModelName.AdvectionEquation_Downwind: AdvectionModelDownwindScheme,
    ModelName.AdvectionEquation_LaxFriedrichs: AdvectionModelLaxFriedrichs,
    ModelName.AdvectionEquation_LaxWendroff: AdvectionModelLaxWendroff,
    ModelName.HeatEquation_ForwardEuler: HeatModelForwardEuler,
    ModelName.HeatEquation_LeapFrog: HeatModelLeapFrog
}


def create_model(model_name):
    if model_name in _models_names_to_objects:
        return _models_names_to_objects[model_name]
    else:
        raise NotImplementedError("Model name {0} is NOT implemented".format(model_name))
