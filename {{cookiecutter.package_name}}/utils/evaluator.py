#!/usr/bin/env python

import torch
import torch.nn as nn

class Evaluator(object):
    """docstring for Evaluate."""
    def __init__(self, config, data, models):
        super(Evaluator, self).__init__()
        self.config = config
        self.data = data

        # models
        self.model = model

    def eval_reconstruction_loss(self):
        pass

    def evaluate(self, n_epoch):
        pass