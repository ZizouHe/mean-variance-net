#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 7 15:24:06 2017

@author: Zizou
train simulation data on alpha_beta_net
"""
import numpy as np
import tensorflow as tf
from data_generate import simulate_data
from alpha_beta_net import alpha_beta_net

data = simulate_data(path="./simulation_data",
                     image_file="simulation_images.npz",
                     label_file="simulation_labels.npz")
abnn = alpha_beta_net(data=data)
abnn.train_net(training_iters=40000, learning_rate=0.01,
               batch_size=1000, display_step=10)
