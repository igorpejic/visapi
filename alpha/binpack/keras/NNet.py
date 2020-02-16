import argparse
import os
import shutil
import time
import random
import numpy as np
import math
from dotdict import *
import sys
sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .KerasBinpackNNet import KerasBinpackNNet as onnet
from keras.models import load_model

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 15,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, predict_move_index=True):
        self.nnet = onnet(game, args, predict_move_index=predict_move_index)
        self.board_x, self.board_y, self.channels = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.predict_v = False
        self.predict_move_index = predict_move_index

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        print(args.batch_size)
        if self.predict_v:
            input_boards, target_pis, target_vs = list(zip(*examples))
        else:
            input_boards, target_pis = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        y = [target_pis]
        if self.predict_v:
            target_vs = np.asarray(target_vs)
            y = [target_pis, target_vs]

        print(self.nnet.model.summary())
        print(args.batch_size)
        self.nnet.model.fit(
            x=input_boards, y=y, batch_size=args.batch_size, epochs=args.epochs,
            validation_split=0.2
        )

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        if self.predict_v:
            pi, v = self.nnet.model.predict(board)
            return pi[0], v[0]
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
        self.nnet.model = load_model(filepath)
