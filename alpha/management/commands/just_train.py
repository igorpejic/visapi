from django.core.management.base import BaseCommand
from alpha.Coach import Coach
import math
import numpy as np

from alpha.binpack.BinPackGame import BinPackGame as Game
from utils import *
from dotdict import dotdict
from data_generator import DataGenerator
from solution_checker import SolutionChecker
import random

ORIENTATIONS = 2

args = dotdict({
    'numIters': 8,
    'numEps': 3,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 2,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 40,

})

def gen_state(width, height, n_tiles, dg):
    '''
    get tiles and solution
    '''

    tiles, solution = dg.gen_matrix_instance(n_tiles, width, height, with_solution=True)
    board = np.zeros([height, width, 1])
    state = np.dstack((board, tiles))
    return state, solution

def solution_to_solution_matrix(solution, rows, cols, return_binary_mask=False):
    '''
    transform solution to 2D matrix with 1 only there where the correct tile should be placed, -1 elsewhere
    This is the expected output of the residual network
    '''
    grid = np.ones((rows, cols))
    grid *= 0
    position = solution[2]
    number_of_ones = solution[0] * solution[1]
    grid[position[0]: position[0] + solution[0], position[1]: position[1] + solution[1]] = 1
    if not return_binary_mask:
        grid[position[0]: position[0] + solution[0], position[1]: position[1] + solution[1]] = 1 / number_of_ones

    
    return grid

def pad_tiles_with_zero_matrices(tiles, n_zero_matrices_to_add, rows, cols):
    '''
    add tiles with zero matrices to compensate for tiles which were already placed
    '''

    zero_matrices = np.zeros([rows, cols, n_zero_matrices_to_add])
    return np.dstack((tiles, zero_matrices))

def get_examples(N_EXAMPLES, n_tiles, height, width, dg, from_file=False, return_binary_mask=False, predict_v=False):
    i = 0
    examples = []
    while i < N_EXAMPLES:
        print(f'{i}/{N_EXAMPLES}')
        state, solution = gen_state(width, height, n_tiles, dg)
        grid = np.zeros([height, width])
        for solution_index, solution_tile in enumerate(solution):
            randomized_solution_order = np.array(solution)
            np.random.shuffle(randomized_solution_order)

            tiles = dg._transform_instance_to_matrix([x[:ORIENTATIONS] for x in randomized_solution_order[solution_index:]])
            tiles = pad_tiles_with_zero_matrices(tiles,  ORIENTATIONS * n_tiles - tiles.shape[2], width, height)
            pi = solution_to_solution_matrix(solution_tile, cols=width, rows=height, return_binary_mask=False).flatten()
            # v = N_TILES - solution_index
            v = 1
            if solution_index == len(solution) - 1 :
                continue
            state = np.dstack((np.expand_dims(grid, axis=2), tiles))
            example = [state, pi]
            if predict_v:
                example.append(v)
            examples.append(example)

            success, grid = SolutionChecker.place_element_on_grid_given_grid(
                solution_tile[:ORIENTATIONS], solution_tile[2], val=1, grid=grid, cols=width, rows=height
            )

        i += 1
    return examples

def get_best_tile_by_prediction(grid, tiles, prediction, dg):
    '''
    1. mask invalid moves
    2. renormalize the probability distribution
    3. for each tile sum up which would be the probability of placing that tile in LFB
    4. return the new grid 

    Returns the tile which is the best in format [(rows, cols), position_to_place]
    '''
    rows, cols = grid.shape
    next_lfb = SolutionChecker.get_next_lfb_on_grid(grid)
    max_probability = -math.inf
    max_index = 0
    best_tile = None
    for i, tile in enumerate(tiles):
        success, _ = SolutionChecker.place_element_on_grid_given_grid(
            tile, next_lfb,
            val=1, grid=grid, cols=cols, rows=rows, get_only_success=True
        )
        if not success:
            continue

        probability = np.sum(
            prediction[next_lfb[0]: next_lfb[0] + tile[0], next_lfb[1]: next_lfb[1] + tile[1]]
        )

        # scale with area
        # probability = probability / (tile[0] * tile[1])

        if probability > max_probability:
            max_index = i
            max_probability = probability
            best_tile = [tile, next_lfb]
    if not best_tile:
        print('No valid tile placement found')
    return best_tile

def get_prediction_masked(prediction, valid_moves):
    inverted_current_board = 1 - valid_moves
    prediction = prediction * inverted_current_board
    # renormalize
    prediction = prediction / np.sum(prediction)
    return prediction

def play_using_prediction(nnet, width, height, n_tiles, dg):
    tiles, grid = dg.gen_tiles_and_board(n_tiles, width, height)

    while True:
        tiles_left = len(tiles)
        if tiles_left == 0:
            print(f"Success: game ended with {tiles_left / ORIENTATIONS} tiles left unplaced.")
            return 0

        tiles_in_matrix_shape = dg._transform_instance_to_matrix(tiles, only_one_orientation=True)
        tiles_in_matrix_shape = pad_tiles_with_zero_matrices(
            tiles_in_matrix_shape, n_tiles * ORIENTATIONS - tiles_left, width, height)

        state = np.dstack((np.expand_dims(grid, axis=2), tiles_in_matrix_shape))
        prediction = nnet.predict(state)
        if len(prediction) == 2:
            prediction, v = prediction


        prediction = np.reshape(prediction, (width, height))

        # get the  probability matrix
        prediction = get_prediction_masked(prediction, state[:, :, 0])

        solution_tile = get_best_tile_by_prediction(grid, tiles, prediction, dg)
        if solution_tile is None:
            print(f"game ended with {tiles_left / ORIENTATIONS} tiles left unplaced.")
            return tiles_left / ORIENTATIONS

        success, grid = SolutionChecker.place_element_on_grid_given_grid(
            solution_tile[0], solution_tile[1], val=1, grid=grid, cols=width,
            rows=height
        )

        if not success:
            print(f"game ended with {tiles_left / ORIENTATIONS} tiles left unplaced.")
            return tiles_left / ORIENTATIONS

        tiles = SolutionChecker.eliminate_pair_tiles(tiles, solution_tile[0])
    return 0


class Command(BaseCommand):

    help = "Run mcts"

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        main()

def main():
    #N_TILES = 8 
    #HEIGHT = 8
    #WIDTH = 8
    N_TILES = 20 
    HEIGHT = 10
    WIDTH = 10
    g = Game(HEIGHT, WIDTH, N_TILES)

    dg = DataGenerator(WIDTH, HEIGHT)

    # from alpha.binpack.tensorflow.NNet import NNetWrapper as nn
    from alpha.binpack.keras.NNet import NNetWrapper as nn
    nnet = nn(g)

    if args.load_model and False:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    n_tiles, height, width = N_TILES, HEIGHT, WIDTH
    state, solution = gen_state(width, height, n_tiles, dg)
    # place tiles one by one
    # generate pair x and y where x is stack of state + tiles
    grid = np.zeros([height, width])
    examples = []
    print('Preparing examples')
    N_EXAMPLES = 800

    _examples = get_examples(N_EXAMPLES, N_TILES, height, width, dg, return_binary_mask=True)

    print(_examples[0][0].shape)
    nnet.train(_examples)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    N_EXAMPLES = 2
    _examples = get_examples(N_EXAMPLES, N_TILES, height, width, dg, from_file=True, return_binary_mask=True)
    for example in _examples:
        prediction = nnet.predict(example[0])
        _prediction = np.reshape(prediction, (width, height))
        _prediction = get_prediction_masked(_prediction, example[0][:, :, 0])
        expected = np.reshape(example[1], (width, height))
        print('-' * 50)
        print('grid state')
        print(example[0][:, :, 0])
        print('expected')
        print(expected)
        print('prediction')
        print(_prediction)

    tiles_left = []
    for i in range(20):
        tiles_left.append(play_using_prediction(nnet, width, height, N_TILES, dg))
        # [0, 6, 4, 2, 2, 2, 0, 4, 4, 8, 6, 2, 2, 6, 6, 8, 6, 4, 4, 4]
    print(tiles_left)

if __name__ == "__main__":
    main()
