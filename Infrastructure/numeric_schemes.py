import numpy as np


class _ModelTemplate(object):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func, nonhomogeneous_term):
        self.x_samples = np.linspace(first_x, last_x, n + 1)
        self.t_samples = np.arange(0, last_t, dt)
        self._dx = self.x_samples[1] - self.x_samples[0]
        self._dt = dt
        self._n = n
        self._current_state = starting_condition_func(self.x_samples)
        self._nonhomogeneous_term = nonhomogeneous_term

    def make_step(self):
        raise NotImplementedError("Each model MUST implement this method!")


class AdvectionEqForwardEuler(_ModelTemplate):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func):
        nonhomogeneous_term = 0  # The equation is assumed to be homogeneous.
        super(AdvectionEqForwardEuler, self).__init__(n, dt, last_t, first_x, last_x, starting_condition_func,
                                                      nonhomogeneous_term)

        self._ratio = self._dt / (2 * self._dx)

    def make_step(self):
        res = []
        i1 = 0
        i2 = 1
        i3 = self._n
        k = 0
        for _ in range(self._n + 1):
            resk = self._current_state[i1] + self._ratio * self._current_state[i2] - self._ratio * self._current_state[i3]
            i1 += 1
            i2 += 1
            i3 += 1
            if k == 0:
                i3 = 0
            elif k == self._n - 1:
                i2 = 0
            k += 1
            res.append(resk)
        self._current_state = np.array(res)
        return self._current_state


class AdvectionModelLeapFrog(_ModelTemplate):
    def __init__(self, n, dt, last_t, first_x, last_x, starting_condition_func):
        nonhomogeneous_term = 0  # The equation is assumed to be homogeneous.
        super(AdvectionModelLeapFrog, self).__init__(n, dt, last_t, first_x, last_x, starting_condition_func,
                                                      nonhomogeneous_term)
        self._forward_euler_first_step_model = AdvectionEqForwardEuler(n, dt, last_t, first_x, last_x,
                                                                       starting_condition_func)

        self._ratio = self._dt / (2 * self._dx)
