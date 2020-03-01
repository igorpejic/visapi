import argparse
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
from .ScalarKerasBinpackNNet import ScalarKerasBinpackNNet
from keras.models import load_model

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'lr': 0.001,
    'dropout': 0.5,
    'epochs': 20,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, predict_move_index=True, scalar_tiles=True, predict_v=False):
        if scalar_tiles:
            self.nnet = ScalarKerasBinpackNNet(game, args, predict_move_index=predict_move_index, scalar_tiles=scalar_tiles, predict_v=predict_v)
        else:
            self.nnet = onnet(game, args, predict_move_index=predict_move_index, scalar_tiles=scalar_tiles)
        self.board_x, self.board_y, self.channels = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.predict_v = predict_v
        self.predict_move_index = predict_move_index
        self.scalar_tiles = scalar_tiles

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        print(args.batch_size)
        if self.predict_v:
            if self.scalar_tiles:
                input_boards, input_tiles, target_pis, target_vs = list(zip(*examples))
                input_tiles = np.asarray(input_tiles)
            else:
                input_boards, target_pis, target_vs = list(zip(*examples))
        else:
            if self.scalar_tiles:
                input_boards, input_tiles, target_pis = list(zip(*examples))
                input_tiles = np.asarray(input_tiles)
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
        kwargs = dict(
            x=input_boards, y=y, batch_size=args.batch_size, epochs=args.epochs,
            validation_split=0.2
        )
        if self.scalar_tiles:
            kwargs['x'] = [input_boards.squeeze(), input_tiles]
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
            board = board[np.newaxis, :, :]
            tiles = tiles[np.newaxis, :, :]
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
        self.nnet.model = load_model(filepath)
