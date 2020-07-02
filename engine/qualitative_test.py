"""##  5. Test"""
from config_parser import get_config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
import random
from actor import Actor
from tqdm import tqdm
from data_generator import DataGenerator
from solution_checker import SolutionChecker
config, _, dir_ = get_config()

config.is_training = False
config.temperature = 1.2 ##### #####

tf.reset_default_graph()
actor = Actor(config) # Build graph

variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name] # Save & restore all the variables.
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

with tf.Session() as sess:  # start session
    sess.run(tf.global_variables_initializer()) # Run initialize op
    
    save_path = "save/"+dir_
    saver.restore(sess, save_path+"/actor.ckpt") # Restore variables from disk.
    
    predictions_length, predictions_length_w2opt = [], []
    for i in tqdm(range(10)): # test instance
        seed_ = 1+random.randint(0, 10000)
        dg =  DataGenerator() # Create Data Generator
        input_batch = dg.train_batch(
            1, actor.n, actor.w, actor.h, actor.dimension,
            seed=i,
            freeze_first_batch=config.freeze_first_batch
        )
        feed = {actor.input_: input_batch} # Get feed dict
        tour, reward = sess.run([actor.tour, actor.reward], feed_dict=feed) # sample tours

        j = np.argmin(reward) # find best solution
        best_permutation = tour[j][:-1]
        predictions_length.append(reward[j])

        solution_checker = SolutionChecker(actor.n, actor.w, actor.h)
        # TODO: find how this is called in numpy (sort by index)
        bins = []
        for el in best_permutation:
            bins.append(input_batch[0][el])

        solution_checker.get_reward(bins)
        grid = solution_checker.grid
        print('reward',reward[j])
        solution_checker.visualize_grid()

        #dataset.visualize_2D_trip(input_batch[0][best_permutation])
        #dataset.visualize_sampling(tour)
        
        # dataset.visualize_2D_trip(opt_tour)
        
    predictions_length = np.asarray(predictions_length) # average tour length
    predictions_length_w2opt = np.asarray(predictions_length_w2opt)
    print("Testing COMPLETED ! Mean length1:",np.mean(predictions_length), "Mean length2:",np.mean(predictions_length_w2opt))

    n1, bins1, patches1 = plt.hist(predictions_length, 50, facecolor='b', alpha=0.75) # Histogram
    n2, bins2, patches2 = plt.hist(predictions_length_w2opt, 50, facecolor='g', alpha=0.75) # Histogram
    plt.xlabel('Tour length')
    plt.ylabel('Counts')
    plt.axis([3., 9., 0, 250])
    plt.grid(True)
    #plt.show()
