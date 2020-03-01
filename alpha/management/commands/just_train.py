import pickle
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
from collections import defaultdict

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

tiles_to_np_array = SolutionChecker.tiles_to_np_array

def gen_state(width, height, n_tiles, dg, scalar_tiles=False):
    '''
    get tiles and solution
    '''

    board = np.zeros([height, width, 1])
    if scalar_tiles:
        tiles, solution = dg.gen_instance(n_tiles, width, height)
        return board, tiles, solution
    else:
        tiles, solution = dg.gen_matrix_instance(n_tiles, width, height, with_solution=True)
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


def one_hot_encode(tiles_ints, solution_tile, n_actions):
    b = np.zeros((n_actions))
    for i, tile in enumerate(tiles_ints):
        if tile == solution_tile:
            b[i] = 1
    return b

get_tiles_with_orientation = SolutionChecker.get_tiles_with_orientation


def get_examples(given_examples, n_tiles, height, width, dg, from_file=False,
                 return_binary_mask=False, predict_v=False, predict_move_index=True,
                 scalar_tiles=False
                 ):
    examples = []
    for i, _example in enumerate(given_examples):
        print(f'{i}/{len(given_examples)}')
        if scalar_tiles:
            state, tiles, solution = _example
        else:
            state, solution = _example
        grid = np.zeros([height, width])
        for solution_index, solution_tile in enumerate(solution):
            solution_copy = np.copy(solution)
            solution_order = np.array(solution_copy[solution_index:])
            solution_tile_dims = solution_tile[:2]

            _tiles_ints = [x[:ORIENTATIONS] for x in solution_order]
            _tiles_ints = get_tiles_with_orientation(_tiles_ints)
            _tiles_ints = SolutionChecker.get_possible_tile_actions_given_grid(grid, _tiles_ints)
            np.random.shuffle(_tiles_ints)
            if scalar_tiles:
                tiles = tiles_to_np_array(SolutionChecker.pad_tiles_with_zero_scalars(
                    _tiles_ints, ORIENTATIONS * n_tiles - len(_tiles_ints)))
                state = state.squeeze()
            else:
                tiles = dg._transform_instance_to_matrix(_tiles_ints, only_one_orientation=True)
                tiles = pad_tiles_with_zero_matrices(tiles, ORIENTATIONS * n_tiles - tiles.shape[2], width, height)
                state = np.dstack((np.expand_dims(grid, axis=2), tiles))
            pi = solution_to_solution_matrix(solution_tile, cols=width, rows=height, return_binary_mask=False).flatten()

            # v = N_TILES - solution_index
            v = 1
            if solution_index == len(solution) - 1 :
                continue

            if predict_move_index:
                n_possible_tiles = SolutionChecker.get_n_nonplaced_tiles(_tiles_ints)
                if n_possible_tiles == 1: # if only one action-tile placement is possible
                    pass
                else:
                    solution_index = _tiles_ints.index(solution_tile_dims)
                    if scalar_tiles:
                        example = [grid.copy(), tiles, one_hot_encode(_tiles_ints, solution_tile_dims, len(tiles) )]
                    else:
                        example = [state, one_hot_encode(_tiles_ints, solution_tile_dims, state.shape[2] - 1)]
                    examples.append(example)
                    # print(_tiles_ints, solution_tile_dims, one_hot_encode(_tiles_ints, solution_tile_dims, state))
            else:
                example = [state, pi]
                if predict_v:
                    example.append(v)

                examples.append(example)

            success, grid = SolutionChecker.place_element_on_grid_given_grid(
                solution_tile[:ORIENTATIONS], solution_tile[2], val=1, grid=grid, cols=width, rows=height
            )

    return examples

def get_best_tile_by_prediction(grid, tiles, prediction, dg, predict_move_index=True):
    '''
    1. mask invalid moves
    2. renormalize the probability distribution
    3. for each tile sum up which would be the probability of placing that tile in LFB
    4. return the new grid 

    Returns the tile which is the best in format [(rows, cols), position_to_place]
    '''
    next_lfb = SolutionChecker.get_next_lfb_on_grid(grid)
    max_probability = -math.inf
    max_index = 0
    best_tile = None
    rows, cols = grid.shape

    tile_probabilities = defaultdict(int)
    tile_counts = defaultdict(int)
    tiles_to_iterate_on = tiles

    for i, tile in enumerate(tiles_to_iterate_on):
        tile = tuple(tile)
        success, _ = SolutionChecker.place_element_on_grid_given_grid(
            tile, next_lfb,
            val=1, grid=grid, cols=cols, rows=rows, get_only_success=True
        )
        if not success:
            continue

        if predict_move_index:
            '''
            The best tile is predicted by taking the sum of the tiles
            with the same (width, height)
            '''
            if tuple(tile) == (0, 0):
                continue
            tile_probabilities[tile] += prediction[i]
            tile_counts[tile] += 1
        else:
            probability = np.sum(
                prediction[next_lfb[0]: next_lfb[0] + tile[0], next_lfb[1]: next_lfb[1] + tile[1]]
            )

            # scale with area
            # probability = probability / (tile[0] * tile[1])

            if probability > max_probability:
                max_index = i
                max_probability = probability
                best_tile = [tile, next_lfb]

    if predict_move_index:
        max_tile = None
        max_val = -math.inf
        for k in tile_probabilities.keys():
            v = tile_probabilities[k] / tile_counts[k]
            if True:
                if v > max_val:
                    max_tile = k
                    max_val = v
            else:
                if k[0] * k[1] > max_val:
                    max_tile = k
                    max_val = k[0] * k[1]
        if max_tile:
            best_tile = [max_tile, next_lfb]

    # print(tile_probabilities)
    # print(tile_counts)
    if not best_tile:
        print('No valid tile placement found')
    return best_tile

def get_prediction_masked(prediction, valid_moves):
    inverted_current_board = 1 - valid_moves
    prediction = prediction * inverted_current_board
    # renormalize
    prediction = prediction / np.sum(prediction)
    return prediction

def state_to_tiles_dims(state, dg):
    tiles = []
    for i in range(state.shape[2] - 1):
        tiles.append(dg.get_matrix_tile_dims(state[:, :, i + 1]))
    return tiles


def play_using_prediction(nnet, width, height, tiles, grid, n_tiles, dg,
                          predict_move_index=False, verbose=False, scalar_tiles=False):
    if scalar_tiles:
        tiles = tiles
    else:
        tiles = state_to_tiles_dims(tiles, dg)
    while True:
        tiles_left = len(tiles)
        if tiles_left == 0:
            print(f"Success: game ended with {tiles_left / ORIENTATIONS} tiles left unplaced.")
            return 0

        _tiles_ints = SolutionChecker.get_possible_tile_actions_given_grid(grid, tiles)
        if len(_tiles_ints) == 0 and tiles_left:
            print(f"game ended with {tiles_left / ORIENTATIONS} tiles left unplaced.")
            return tiles_left / ORIENTATIONS
        np.random.shuffle(_tiles_ints)
        if scalar_tiles:
            _tiles = tiles_to_np_array(SolutionChecker.pad_tiles_with_zero_scalars(
                _tiles_ints, ORIENTATIONS * n_tiles - len(_tiles_ints), width, height))
            # state = state.squeeze()

            prediction = nnet.predict([grid, tiles_to_np_array(_tiles)])
        else:
            tiles_in_matrix_shape = dg._transform_instance_to_matrix(_tiles_ints, only_one_orientation=True)
            tiles_in_matrix_shape = pad_tiles_with_zero_matrices(
                tiles_in_matrix_shape, n_tiles * ORIENTATIONS - tiles_in_matrix_shape.shape[2], width, height)

            state = np.dstack((np.expand_dims(grid, axis=2), tiles_in_matrix_shape))
            prediction = nnet.predict(state)

        if not predict_move_index:
            if len(prediction) == 2: # if we are also predicting v
                prediction, v = prediction

            prediction = np.reshape(prediction, (width, height))

            # get the  probability matrix
            prediction = get_prediction_masked(prediction, state[:, :, 0])

        if verbose:
            print('-' * 50)
            print(grid, prediction)
            print(tiles)
        if scalar_tiles:
            _ttiles = tiles
        else:
            _ttiles = state_to_tiles_dims(state, dg)
        solution_tile = get_best_tile_by_prediction(
            grid, _ttiles, prediction, dg, predict_move_index=predict_move_index)
        if verbose:
            print(solution_tile)
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

        if scalar_tiles:
            tiles = [tuple(x) for x in tiles]
        tiles = SolutionChecker.eliminate_pair_tiles(tiles, solution_tile[0])
    return 0


class Command(BaseCommand):

    help = "Run mcts"

    def add_arguments(self, parser):
        parser.add_argument('--load_model', action='store_true', help='Load pretrained model')
        parser.add_argument('--load_examples', action='store_true', help='Load train data which was once generated')

    def handle(self, *args, **options):
        main(options)

def count_n_of_non_placed_tiles(tiles):
    return len([tile for tile in tiles if tuple(tile) != (0, 0)])

def get_n_examples(N_EXAMPLES, width, height, n_tiles, dg, scalar_tiles=True):
    examples = []
    for i in range(N_EXAMPLES):
        print(f'{i}/{N_EXAMPLES}')
        example = gen_state(width, height, n_tiles, dg, scalar_tiles=scalar_tiles)
        examples.append([*example])
    return examples

def main(options):
    #N_TILES = 8 
    #HEIGHT = 8
    #WIDTH = 8
    SCALAR_TILES = True
    predict_move_index = True
    N_TILES = 10 
    HEIGHT = 10
    WIDTH = 10
    g = Game(HEIGHT, WIDTH, N_TILES)

    dg = DataGenerator(WIDTH, HEIGHT)

    # from alpha.binpack.tensorflow.NNet import NNetWrapper as nn
    from alpha.binpack.keras.NNet import NNetWrapper as nn
    nnet = nn(g, scalar_tiles=SCALAR_TILES)

    n_tiles, height, width = N_TILES, HEIGHT, WIDTH
    if options['load_model']:
        nnet.load_checkpoint()
    else:
        # place tiles one by one
        # generate pair x and y where x is stack of state + tiles
        print('Preparing examples')
        N_EXAMPLES = 1700

        examples = get_n_examples(N_EXAMPLES, width, height, n_tiles, dg, scalar_tiles=SCALAR_TILES)
        print(examples)
        if options['load_examples']:
            with open('models/train_examples.pickle', 'rb') as f:
                train_examples = pickle.load(f)
        else:
            train_examples = get_examples(
                examples, N_TILES, height, width, dg, return_binary_mask=True,
                predict_move_index=True, scalar_tiles=SCALAR_TILES)
            with open('models/train_examples.pickle', 'wb') as f:
                pickle.dump(train_examples, f)
        nnet.train(train_examples)
        nnet.save_checkpoint()

    np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)}, linewidth=115)

    N_EXAMPLES = 20
    examples = get_n_examples(N_EXAMPLES, width, height, n_tiles, dg, scalar_tiles=SCALAR_TILES)
    _examples = get_examples(examples, N_TILES, height, width, dg, from_file=False, return_binary_mask=True, predict_move_index=True, scalar_tiles=SCALAR_TILES)[:N_EXAMPLES]
    total_correct = 0
    total_count = 0
    
    n_empty_tiles_with_fails = [0] * (N_TILES + 1)
    for example in _examples:
        prediction = nnet.predict([example[0], example[1]])
        if predict_move_index:
            _prediction = prediction
            max_index = np.argmax(prediction)
            _prediction_index = max_index
            if SCALAR_TILES:
                expected = np.argmax(example[2])
            else:
                expected = np.argmax(example[1])

            print('-' * 50)
            # if not scalar_tiles:
               # print('grid state')
               # print(example[0][:, :, 0])
               # print(state_to_tiles_dims(example[0], dg))
            # print('expected')
            if SCALAR_TILES:
                expected_tile = example[1][expected]
                prediction_tile = example[1][_prediction_index]
                print(example[1].tolist())
            else:
                expected_tile = dg.get_matrix_tile_dims(example[0][:, :, expected + 1])
                prediction_tile = dg.get_matrix_tile_dims(example[0][:, :, _prediction_index + 1])
            print(example[0])
            # print(expected, expected_tile)
            #print('prediction')
            # print(_prediction)
            #print(_prediction_index, prediction_tile)
            print(f'expected: {expected_tile}, got: {prediction_tile}')
            if SCALAR_TILES:
                if np.array_equal(expected_tile, prediction_tile):
                    total_correct += 1
                else:
                    n_empty_tiles_with_fails[count_n_of_non_placed_tiles(example[1]) // 2] += 1
            else:
                if expected_tile == prediction_tile:
                    total_correct += 1
                else:
                    n_empty_tiles_with_fails[count_n_of_non_placed_tiles(state_to_tiles_dims(example[0], dg)) // 2] += 1
            total_count += 1
        else:
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

    if predict_move_index:
        print(f'In total guessed: {total_correct}/{total_count} = {100*(total_correct/ total_count)}%')
        print(n_empty_tiles_with_fails)

        print('-' * 100)

    tiles_left = []
    for example in examples:
        if SCALAR_TILES:
            state, tiles, _ = example
            tiles = get_tiles_with_orientation(tiles.tolist())
        else:
            tiles, _ = example
        # tiles = dg.get_matrix_tile_dims(tiles)
        grid = np.zeros((width, height))
        tiles_left.append(play_using_prediction(
            nnet, width, height, tiles, grid, N_TILES, dg, predict_move_index, scalar_tiles=SCALAR_TILES))
        # [0, 6, 4, 2, 2, 2, 0, 4, 4, 8, 6, 2, 2, 6, 6, 8, 6, 4, 4, 4]
    print(tiles_left)

if __name__ == "__main__":
    main()
