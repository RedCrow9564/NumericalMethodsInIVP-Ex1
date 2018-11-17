from enum import Enum


class ConfigParams(Enum):
    method_name = 'method name'
    dx = 'dx'
    lamda = 'lamda'


class DtInitializerMethod(Enum):
    linear = 'linear'
    square = 'square'


def _linear_dt_init(init_config):
    lamda = init_config[ConfigParams.lamda]
    dx = init_config[ConfigParams.dx]
    dt = lamda * dx
    return dt


def _square_dt_init(init_config):
    lamda = init_config[ConfigParams.lamda]
    dx = init_config[ConfigParams.dx]
    dt = lamda * dx ** 2
    return dt


_dt_initializer_to_method = {
    DtInitializerMethod.linear: _linear_dt_init,
    DtInitializerMethod.square: _square_dt_init
}


def calc_dt(init_config):
    init_name = init_config[ConfigParams.method_name]
    if init_name in _dt_initializer_to_method:
        return _dt_initializer_to_method[init_name](init_config)
    else:
        raise NotImplementedError('No such dt initializer method {0}'.format(init_name))
