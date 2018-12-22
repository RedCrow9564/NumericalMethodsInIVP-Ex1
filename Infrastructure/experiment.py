import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from Infrastructure.dt_initializer import calc_dt, ConfigParams
from Infrastructure.numeric_schemes import create_model


class Experiment(object):
    def __init__(self, model_name, first_t, last_t, dt, first_x, last_x, dx, n, exact_solution, nonhomogeneous_term):
        self._experiment_time_steps_num = int(np.floor(last_t / dt)) - 1
        self._last_calculated_time = np.floor(last_t / dt) * dt
        self._last_t = last_t
        self._exact_solution = exact_solution
        self._x_values = np.arange(first_x, last_x, dx)

        selected_model_type = create_model(model_name)
        start_conditions = lambda x: exact_solution(x, 0)
        self.model = selected_model_type(n, dt, first_t, first_x, last_x,
                                         starting_condition_func=start_conditions,
                                         nonhomogeneous_term=nonhomogeneous_term)
        self.results = None
        self.model_error = None

    def run(self):
        for _ in range(self._experiment_time_steps_num):
            self.model.make_step()
        self.results = self.model.current_state

        exact_values = self._exact_solution(self._x_values, self._last_calculated_time)
        self.model_error = np.linalg.norm(exact_values - self.results)

        return self.results, self.model_error

    def plot_results(self):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        exact_values = self._exact_solution(self._x_values, self._last_calculated_time)
        plt.plot(self._x_values, self.results, label='Approximation')
        plt.plot(self._x_values, exact_values, label='Exact')
        plt.legend()
        plt.show()


class SingleNManyLambdasExperiments(object):
    def __init__(self, model_name, n, lamda_list, first_t, last_t, first_x, last_x, dt_init_method, exact_solution,
                 nonhomogeneous_term):
        self._n = n
        self._dx = (last_x - first_x) / (n + 1)
        self._dt_init_method = dt_init_method
        self._model_errors = []
        self._model_name = model_name
        self._lambda_list = lamda_list
        self._experiment_template = lambda dt, dx, n: Experiment(model_name, first_t, last_t, dt, first_x, last_x, dx,
                                                                 n, exact_solution, nonhomogeneous_term)

    def _create_experiment(self, lamda, n):
        dt_init_config = {
            ConfigParams.method_name: self._dt_init_method,
            ConfigParams.lamda: lamda,
            ConfigParams.dx: self._dx
        }
        dt = calc_dt(dt_init_config)
        return self._experiment_template(dt, self._dx, n)

    def run_experiments(self):
        for lamda in self._lambda_list:
            experiment = self._create_experiment(lamda, self._n)
            _, experiment_error = experiment.run()
            #experiment.plot_results()
            self._model_errors.append(experiment_error)

    def plot_results(self):
        self._model_errors = np.array(self._model_errors)
        np.place(self._model_errors, self._model_errors == 0, np.finfo(np.float32).eps)
        self._model_errors = self._model_errors.tolist()

        weighed_errors = np.log10(np.array(self._model_errors).dot(np.sqrt(self._dx)))
        flipped_lambdas = np.log10(self._lambda_list)

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax = plt.subplot('111')
        ax.plot(flipped_lambdas, weighed_errors)
        ax.set_title(r'{0} scheme for N = {1}'.format(self._model_name.value, self._n), fontsize=14)
        ax.set_xlabel(r'log($\lambda$)', fontsize=14)
        ax.set_ylabel(r'\textit{Weighed L2 approximation log error}', fontsize=14)

        plt.scatter(flipped_lambdas, weighed_errors)
        plt.show()


class SingleLambdaManyNExperiments(object):
    def __init__(self, model_name, n_list, lamda, first_t, last_t, first_x, last_x, dt_init_method, exact_solution,
                 nonhomogeneous_term):
        self._n_list = n_list
        self._dx_list = ((last_x - first_x) / (n_list + 1)).tolist()
        self._dt_init_method = dt_init_method
        self._model_name = model_name
        self._model_errors = []
        self._lambda = lamda
        self._experiment_template = lambda dt, dx, n: Experiment(model_name, first_t, last_t, dt, first_x, last_x, dx,
                                                                 n, exact_solution, nonhomogeneous_term)

    def _create_experiment(self, dx, n):
        dt_init_config = {
            ConfigParams.method_name: self._dt_init_method,
            ConfigParams.lamda: self._lambda,
            ConfigParams.dx: dx
        }
        dt = calc_dt(dt_init_config)
        return self._experiment_template(dt, dx, n)

    def run_experiments(self):
        for dx, n in zip(self._dx_list, self._n_list):
            _, experiment_error = self._create_experiment(dx, n).run()
            self._model_errors.append(experiment_error)

    def plot_results(self):
        self._model_errors = np.array(self._model_errors)
        np.place(self._model_errors, self._model_errors == 0, np.finfo(np.float32).eps)
        self._model_errors = self._model_errors.tolist()
        weighed_errors = np.log10(np.multiply(np.sqrt(self._dx_list), self._model_errors))
        flipped_dx = np.log10(self._dx_list)
        try:
            slope, c = np.polyfit(flipped_dx, weighed_errors, deg=1)
        except np.linalg.LinAlgError:
            slope = np.nan
            c = np.nan

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax = plt.subplot('111')
        ax.plot(flipped_dx, c + slope * flipped_dx)
        ax.set_title(r'{0} scheme for $\lambda$ = {1}'.format(self._model_name.value, self._lambda), fontsize=14)
        ax.set_xlabel(r'\textit{log(dx)}', fontsize=14)
        ax.set_ylabel(r'\textit{Weighed L2 approximation log error}', fontsize=14)
        at = AnchoredText("slope = {0}".format(slope), prop=dict(size=12), frameon=True, loc=2)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        plt.scatter(flipped_dx, weighed_errors)
        plt.show()
