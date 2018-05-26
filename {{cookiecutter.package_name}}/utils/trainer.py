#!/usr/bin/env python

import torch
import torch.nn as nn

class Trainer(object):
    def __init__(self, config, data, model):
        super(Trainer, self).__init__()
        self.config = config
        self.data = data

        self.model = model

        # training stats
        self.stats = {}
        self.stats['vae_training_logs'] = []
        self.stats['lat_dis_training_logs'] = []
        self.stats['ptc_dis_training_logs'] = []

        # best reconstruction loss / best accuracy
        self.best_loss = 1e12
        self.best_accu = -1e12
        self.n_total_iter = 0

    def step(self):
        pass

    def save_checkpoint(self):
        pass
    
    def save_model(self, name):
        pass

    def save_best_periodic(self, to_log):
        pass
    
    def adjust_learning_rate(self):
        pass
