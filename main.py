import numpy as np

from Infrastructure.numeric_schemes import ModelName
from Infrastructure.experiment import SingleLambdaManyNExperiments, SingleNManyLambdasExperiments
from Infrastructure.dt_initializer import DtInitializerMethod


def exact_solution(x, t):
    return np.cos(2 * np.pi * (x + t))


def nonhomogeneous_term_heat_model(x, t):
    return -2 * np.pi * np.sin(2 * np.pi * (x + t)) + \
           4 * np.pi ** 2 * np.cos(2 * np.pi * (x + t))


def find_best_lambda(model_name, lambdas, dt_init_method, first_t, last_t, first_x, last_x, n, exact_solution,
                     nonhomogeneous_term):
    experiments = SingleNManyLambdasExperiments(model_name, n, lambdas, first_t=first_t, last_t=last_t, first_x=first_x,
                                                last_x=last_x, dt_init_method=dt_init_method,
                                                exact_solution=exact_solution,
                                                nonhomogeneous_term=nonhomogeneous_term)
    experiments.run_experiments()
    experiments.plot_results()


def find_approximation_rate(model_name, lamda, dt_init_method, first_t, last_t, first_x, last_x, n_list, exact_solution,
                            nonhomogeneous_term):
    experiments = SingleLambdaManyNExperiments(model_name, n_list, lamda, first_t=first_t, last_t=last_t,
                                               first_x=first_x, last_x=last_x, dt_init_method=dt_init_method,
                                               exact_solution=exact_solution,
                                               nonhomogeneous_term=nonhomogeneous_term)
    experiments.run_experiments()
    experiments.plot_results()


def main():
    first_x = 0
    last_x = 1
    first_t = 0
    last_t = 5

    lambdas_list = np.arange(1, 5, 1).tolist()
    first_n = 16
    nonhomogeneous_term = nonhomogeneous_term_heat_model  # lambda x, t: np.zeros(x.shape)
    model_name = ModelName.HeatEquation_LeapFrog
    dt_init_method = DtInitializerMethod.square

    find_best_lambda(model_name, lambdas_list, dt_init_method, first_t, last_t, first_x, last_x, first_n,
                     exact_solution, nonhomogeneous_term)

    best_lambda = 4
    n_list = np.power(2, list(range(4, 9)))

    find_approximation_rate(model_name, best_lambda, dt_init_method, first_t, last_t, first_x, last_x, n_list,
                            exact_solution, nonhomogeneous_term)


if __name__ == '__main__':
    main()
