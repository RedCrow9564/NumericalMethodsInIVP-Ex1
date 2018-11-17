import numpy as np

from Infrastructure.numeric_schemes import ModelName
from Infrastructure.experiment import SingleLambdaManyNExperiments, SingleNManyLambdasExperiments
from Infrastructure.dt_initializer import DtInitializerMethod


def exact_solution(x, t):
    return np.cos(2 * np.pi * (x + t))


def start_conditions(x):
    return exact_solution(x, 0)


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

    lambdas_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] + np.arange(1, 17.5, 0.5).tolist()
    first_n = 16
    nonhomogeneous_term = lambda y, t: np.zeros(y.shape)
    model_name = ModelName.AdvectionEquation_LeapFrog
    dt_init_method = DtInitializerMethod.square

    find_best_lambda(model_name, lambdas_list, dt_init_method, first_t, last_t, first_x, last_x, first_n,
                     exact_solution, nonhomogeneous_term)

    # best_lambda = 1
    # n_list = np.power(2, list(range(4, 7)))
    #
    # find_approximation_rate(model_name, best_lambda, dt_init_method, first_t, last_t, first_x, last_x, n_list,
    #                         exact_solution, nonhomogeneous_term)


if __name__ == '__main__':
    main()
