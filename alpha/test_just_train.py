import unittest
from MCTS import CustomMCTS, State, render_to_dict, eliminate_pair_tiles, get_max_index
from data_generator import DataGenerator
from just_train import get_best_tile_by_prediction

import numpy as np

class TestGetBestActionBasedOnPrediction(unittest.TestCase):

    def test_initialize_state(self):
        w = 3
        h = 3
        dg = DataGenerator(w, h)
        prediction = np.array([
            [0, 0,   0.3],
            [0, 0,   0.4],
            [0, 0.1, 0.1]
        ])

        grid = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ])

        tiles = [(1, 1), (1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]

        self.assertEqual(len(tiles), 6)
        ret = get_best_tile_by_prediction(grid, tiles, prediction, dg)
        self.assertEqual(ret, [(2, 1), (0, 2)])

    def test_initialize_state_2(self):
        w = 3
        h = 3
        dg = DataGenerator(w, h)
        prediction = np.array([
            [0, 0,   0.6],
            [0, 0,   0.4],
            [0, 0.1, 0.1]
        ])

        grid = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ])

        tiles = [(1, 1), (1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]

        ret = get_best_tile_by_prediction(grid, tiles, prediction, dg)
        self.assertEqual(ret, [(1, 1), (0, 2)])

    def test_initialize_state_3(self):
        w = 3
        h = 3
        dg = DataGenerator(w, h)
        prediction = np.array([
            [10, 10,   0.6],
            [10, 10,   0.8],
            [1.8, 1.9, 0.71]
        ])

        grid = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]
        ])

        tiles = [(1, 1), (1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]

        ret = get_best_tile_by_prediction(grid, tiles, prediction, dg)
        self.assertEqual(ret, [(3, 1), (0, 2)])
