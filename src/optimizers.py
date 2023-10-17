#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Optimizers are defined in this file.

"""

from torch import optim


def adam(params, learning_rate):
    return optim.Adam(params=params,
                      lr=learning_rate,
                      betas=(0.9, 0.999),
                      eps=1e-8,
                      weight_decay=0,
                      amsgrad=False,
                      foreach=None,
                      maximize=False,
                      capturable=False)


def sgd(params, learning_rate):
    return optim.SGD(params=params, 
                     lr=learning_rate,
                     momentum=0,
                     dampening=0,
                     weight_decay=0,
                     nesterov=False,
                     maximize=False,
                     foreach=None)


def adagrad(params, learning_rate):
    return optim.Adagrad(params=params,
                         lr=learning_rate,
                         lr_decay=0,
                         weight_decay=0,
                         initial_accumulator_value=0,
                         eps=1e-10,
                         foreach=None,
                         maximize=False)
    
    
    
   
