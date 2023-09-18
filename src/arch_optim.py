#/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file contains the functions responsible for linking the main file to the architectures and optimizers.    
    
"""
import optimizers as optim
import architectures as arch

architectures = {
    "deeplabv3_resnet101": arch.deeplabv3_resnet101,
    "deeplabv3_resnet50": arch.deeplabv3_resnet50,
    "fcn_resnet50": arch.fcn_resnet50
}

optimizers = {
    "adam": optim.adam,
    "sgd": optim.sgd,
    "adagrad": optim.adagrad,
}

def get_architecture(architecture, in_channels, out_classes, pretrained):
    # Return the model.
    return architectures[architecture.casefold()](in_channels=in_channels,
                                                  out_classes=out_classes,
                                                  pretrained=pretrained)


def get_optimizer(optimizer, model, learning_rate):
    # Return the optimizer.
    return optimizers[optimizer.casefold()](params=model.parameters(),
                                            learning_rate=learning_rate)
