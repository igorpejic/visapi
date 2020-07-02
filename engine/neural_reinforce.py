# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2018 Michel Deudon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

from data_generator import DataGenerator
from solution_checker import SolutionChecker

from tqdm import tqdm
import matplotlib.pyplot as plt
from config_parser import get_config
from actor import Actor

"""## 1. Data Generator"""

dataset = DataGenerator() # Create Data Generator

n = 20
w = 50
h = 50

# input_batch = dataset.test_batch(batch_size=128, n=n, w=w, h=h, dimensions=2, seed=123) # Generate some data
# dataset.visualize_2D_trip(input_batch[0]) # 2D plot for coord batch

"""## 2. Config"""

config, _, dir_ = get_config()

"""## 3. Model"""

tf.reset_default_graph()
actor = Actor(config) # Build graph

variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name] # Save & restore all the variables.
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

with tf.Session() as sess: # start session
    sess.run(tf.global_variables_initializer()) # Run initialize op
    variables_names = [v.name for v in tf.trainable_variables() if 'Adam' not in v.name]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        #print("Variable: ", k, "Shape: ", v.shape) # print all variables
        pass

"""## 4. Train"""

np.random.seed(123) # reproducibility
tf.set_random_seed(123)

#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.Session() as sess: # start session
    sess.run(tf.global_variables_initializer()) # run initialize op
    writer = tf.summary.FileWriter('summary/'+dir_, sess.graph) # summary writer
    
    for i in tqdm(range(config.nb_steps)): # Forward pass & train step
        input_batch = dataset.train_batch(
            actor.batch_size, actor.n, actor.w, actor.h, actor.dimension,
            freeze_first_batch=config.freeze_first_batch
        )
        feed = {actor.input_: input_batch} # get feed dict
        reward, predictions, summary, _, _ = sess.run([actor.reward, actor.predictions, actor.merged, actor.trn_op1, actor.trn_op2], feed_dict=feed)

        if i % 50 == 0: 
            print('reward',np.mean(reward))
            print('predictions',np.mean(predictions))
            writer.add_summary(summary,i)
        
        if i % 100 == 0:
            save_path = "save/"+dir_
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            saver.save(sess, save_path+"/actor.ckpt") # save the variables to disk
    print("Training COMPLETED! Model saved in file: %s" % save_path)
