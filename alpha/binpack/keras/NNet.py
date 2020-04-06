import argparse
import datetime
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .KerasBinpackNNet import KerasBinpackNNet as onnet
from .ScalarKerasBinpackNNet import ScalarKerasBinpackNNet, binary_focal_loss, true_positives, false_positives, custom_accuracy, weighted_cross_entropy
import tensorflow as tf
if hasattr(tf, 'keras'):
    from tensorflow.keras.callbacks import Callback
    import tensorflow.keras as keras
    from tensorflow.keras.models import load_model
    import tensorflow.keras.backend as K
else:
    from keras.models import load_model
    from keras.callbacks import Callback
    import keras

logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=logdir,
    # update_freq='batch',
)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, validation_data, display):

        self.step = 0
        self.display = display
        self.metric_cache = {}
        super(NBatchLogger, self).__init__()
        self.validation_data = validation_data

    def on_batch_end(self, batch, logs={}, **kwargs):

        self.step += 1
        for i, k in enumerate([m for m in self.params['metrics'] if 'val_' not in m]):
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]

        if self.validation_data:
            val_logs = self.model.test_on_batch(*self.validation_data)
        if self.validation_data:
            for i, k in enumerate([m for m in self.params['metrics'] if 'val_' in m]):
                self.metric_cache['val_' + k] = self.metric_cache.get(k, 0) + val_logs[i]

        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                if 'val_' not in k:
                    val = v / self.display
                else:
                    val = v
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
                tf.summary.scalar('batch_'+ k,  val, step=self.step)
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 128,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, predict_move_index=True, scalar_tiles=True, predict_v=False):
        self.input_tiles_individually = False
        if scalar_tiles:
            self.nnet = ScalarKerasBinpackNNet(game, args, predict_move_index=predict_move_index, scalar_tiles=scalar_tiles, predict_v=predict_v, 
                                               individual_tiles = self.input_tiles_individually)
        else:
            self.nnet = onnet(game, args, predict_move_index=predict_move_index, scalar_tiles=scalar_tiles)
        self.board_x, self.board_y, self.channels = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.predict_v = predict_v
        self.predict_move_index = predict_move_index
        self.scalar_tiles = scalar_tiles


    def train(self, examples, validation_data=None):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        if self.predict_v:
            if self.scalar_tiles:
                input_boards, input_tiles, target_pis, target_vs = list(zip(*examples))
                input_tiles = np.asarray(input_tiles, dtype=np.float16)
            else:
                input_boards, target_pis, target_vs = list(zip(*examples))
        else:
            if self.scalar_tiles:
                input_boards, input_tiles, target_pis = list(zip(*examples))
                input_tiles = np.asarray(input_tiles, dtype=np.float16)
                print(input_tiles)
            else:
                input_boards, target_pis = list(zip(*examples))

        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        y = [target_pis.astype(np.float16)]
        if self.predict_v:
            target_vs = np.asarray(target_vs)
            y = [target_pis, target_vs]

        print(self.nnet.model.summary())
        print(args.batch_size)
        kwargs = dict(
            x=input_boards, y=y, batch_size=args.batch_size, epochs=args.epochs,
            validation_split=0.2,
        )
        if validation_data:
            del kwargs['validation_split']
            if self.scalar_tiles:
                val_input_boards, val_input_tiles, val_target_pis = list(zip(*validation_data))
                val_input_tiles = np.asarray(val_input_tiles, dtype=np.float16)

            val_input_boards = np.asarray(val_input_boards)
            val_target_pis = np.asarray(val_target_pis)
            val_y = [val_target_pis]
            kwargs['validation_data'] = (
                [
                 val_input_boards[..., np.newaxis],
                 #val_input_boards.squeeze(),
                 val_input_tiles], (val_y))
            #kwargs['callbacks'] = callbacks=[tensorboard_callback, NBatchLogger(kwargs['validation_data'], 64)]
            kwargs['callbacks'] = callbacks=[tensorboard_callback]
        else:
            kwargs['callbacks'] = callbacks=[tensorboard_callback]

        if self.scalar_tiles:
            if self.input_tiles_individually:
                # print(input_tiles.shape, input_boards.shape)
                kwargs['x'] = [
                    input_boards.squeeze(),
                    input_boards,
                    *[input_tiles[:, x] for x in range(input_tiles.shape[1])]]
            else:
                kwargs['x'] = [
                    input_boards[..., np.newaxis],
                    # input_boards.squeeze(),
                    input_tiles.astype(np.float16)]

            self.nnet.model.fit(
                **kwargs
            )
        else:
            self.nnet.model.fit(
                **kwargs
            )

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        if self.scalar_tiles:
            board, tiles = board
            board = board[np.newaxis, :, :, np.newaxis]
            tiles = tiles[np.newaxis, :, :].astype(np.float16)
        else:
            board = board[np.newaxis, :, :]
        # run
        if self.predict_v:
            if self.scalar_tiles:
                pi, v = self.nnet.model.predict([board, tiles])
            else:
                pi, v = self.nnet.model.predict(board)
            return pi[0], v[0]
        else:
            if self.scalar_tiles:
                pi = self.nnet.model.predict([board, tiles])
            else:
                pi = self.nnet.model.predict(board)
            return pi[0]

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))

    def save_checkpoint(self, folder='models', filename='model.h5'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save(filepath)

    def load_checkpoint(self, folder='models', filename='model.h5'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model = load_model(
            filepath, custom_objects={
                'binary_focal_loss_fixed': binary_focal_loss(),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'custom_accuracy': custom_accuracy,
                'weighted_cross_entropy': weighted_cross_entropy,
                }
        )
