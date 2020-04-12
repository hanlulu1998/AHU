import numpy as np

'''
@Descripttion: 常见优化器
@version: 1.0
@Author: Han Lulu
@Date: 2020-04-11 11:03:00
@LastEditors: Han Lulu
@LastEditTime: 2020-04-11 11:07:22
'''


# SGD优化器
def sgd(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    return w, config


# 带动量的SGD优化器
def sgd_momentum(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    config['velocity'] = v
    return next_w, config


# rmsprop优化器
def rmsprop(x, dx, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    config['cache'] = config['decay_rate'] * config['cache'] + (
            1 - config['decay_rate']) * dx ** 2
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) +
                                                 config['epsilon'])

    return next_x, config


# Adam优化器
def adam(x, dx, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx **
                                                                           2)
    next_x = x - config['learning_rate'] * config['m'] / (
            np.sqrt(config['v']) + config['epsilon'])
    return next_x, config
