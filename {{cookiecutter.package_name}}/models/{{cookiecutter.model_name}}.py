#!/usr/bin/env python

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        