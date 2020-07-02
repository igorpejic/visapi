import numpy as np
import matplotlib.pyplot as plt
import math
import bisect
from sklearn.decomposition import PCA
from data_generator import DataGenerator
# from numba import jit
from state import State
import uuid
import search

from sortedcontainers import SortedKeyList
from collections import OrderedDict


ALL_TILES_USED = 'ALL_TILES_USED'
TILE_CANNOT_BE_PLACED = 'TILE_CANNOT_BE_PLACED'
NO_NEXT_POSITION_TILES_UNUSED = 'NO_NEXT_POSITION_TILES_UNUSED'

GLOBAL_OCCUPIED_VAL = 1

class SolutionChecker(object):


    def __init__(self, n, cols, rows):
        self.n = n
        self.cols = cols
        self.rows = rows
        self.LFBs = SortedKeyList([], key=lambda x: (x[1], x[0]))
        self.LFBs.add((0, 0))
        self.grid = self.initialize_grid()


    def initialize_grid(self):
        return np.array([[0 for x in range(self.cols)] for y in range(self.rows)])


    def get_rewards(self, batch_bins, count_tiles=False, combinatorial_reward=False):
        batch_rewards = []
        # print(batch_bins)
        for  _bin in batch_bins:
            self.grid = self.initialize_grid()
            batch_rewards.append(self.get_reward(_bin, count_tiles=count_tiles, combinatorial_reward=combinatorial_reward))
        # return np.mean(batch_rewards).astype(np.float32)
        return np.array(batch_rewards).astype(np.float32)

    def get_reward(self, bins, count_tiles=False, combinatorial_reward=False):
        '''
        perfect reward is 0 area wasted
        as_tiles_non_placed - reward is given by number of tiles non_placed from the total number of tiles
        combinatorial_reward - if True it will stop after first bin is not placed
        '''

        reward = 0
        bins = bins[:-1]
        for i, _bin in enumerate(bins):
            next_lfb = self.get_next_lfb()
            if not next_lfb:
                break

            placed, new_grid = self.place_element_on_grid(_bin, next_lfb, i + 1, self.cols, self.rows)
            if not placed:
                if combinatorial_reward:
                    return 1
                if count_tiles:
                    reward += 1
                else:
                    reward += (_bin[0] * _bin[1])
            else:
                self.grid = new_grid


        # scale from 0 to 1
        if reward == 0:
            return 0
        else:
            if count_tiles:
                return reward / self.n
            else:
                return reward / (self.cols * self.rows)

    @staticmethod
    def get_next_lfb_on_grid(grid):
        lfb = None
        n_cols = grid.shape[1]
        res = search.find_first(0, grid.ravel()) 
        if res == -1:
            return None
        res_row = res // n_cols
        res_col = res % n_cols

        return (res_row, res_col)

    def get_next_lfb(self):
        return SolutionChecker.get_next_lfb_on_grid(self.grid)

    @staticmethod
    def place_element_on_grid_given_grid(_bin, position, val, grid, cols, rows, get_only_success=False, colorful_states=False):

        if position[1] + _bin[1] > cols:
            # print(f'{position[1] + _bin[1]} bigger than width')
            return False, None
        if position[0] + _bin[0] > rows:
            # print(f'{position[0] + _bin[0]} bigger than height')
            return False, None

        # need to check only the width as height is always free
        slice_of_new_grid_any_one = grid[position[0], position[1]: position[1] + _bin[1]]

        # is any number in slice non-zero ??
        if colorful_states:
            if np.any(slice_of_new_grid_any_one):
                return False, None
            elif get_only_success:
                return True, None
        # is any number in slice 1??
        else:
            if GLOBAL_OCCUPIED_VAL in slice_of_new_grid_any_one:
                return False, None
            elif get_only_success:
                return True, None

        grid[position[0]: position[0] + _bin[0], position[1]: position[1] + _bin[1]] = val
        # for i in range(int(_bin[1])):
        #     for j in range(int(_bin[0])):
        #         row = new_grid[position[1] + i]
        #         if row[position[0] + j] != 0:
        #             # print(f'position ({position[1] + i} {position[0] + j}) already taken')
        #             return False, None
        #         row[position[0] + j] = val

        return True, grid

    def place_element_on_grid(self, _bin, position, val, cols, rows):
        return SolutionChecker.place_element_on_grid_given_grid(_bin, position, val, self.grid, self.cols, self.rows)

    # def get_reward(self, bins):
    #     '''
    #     perfect reward is w * h - 0
    #     '''
    #     reward = self.w * self.rows

    #     bins_processed = []
    #     for _bin in bins:
    #         lfbs_to_add = []
    #         if self.is_bin_outside_borders(_bin):
    #             # TODO
    #             reward -= 1
    #             print(_bin, self.LFBs, 'could not fill')
    #         else:
    #             old_lfb = self.LFBs[0]

    #             left_point = old_lfb[0], old_lfb[1] + _bin[1]
    #             low_right_point = old_lfb[0] + _bin[0], old_lfb[1]
    #             high_right_point = old_lfb[0] + _bin[0], old_lfb[1] + _bin[1]

    #             if left_point[1] == self.rows:  # reached the ceiling
    #                 print('reached the ceiling')
    #                 if high_right_point[0] != self.w:
    #                     lfbs_to_add.extend([low_right_point])
    #             elif high_right_point[0] == self.w:
    #                 lfbs_to_add.extend([left_point])

    #             else:
    #                 lfbs_to_add.extend([left_point, low_right_point, high_right_point])
    #                 #self.LFBs.add(left_point)

    #                 #self.LFBs.add(low_right_point)
    #                 #self.LFBs.add(high_right_point)

    #         # TODO: now check which lfbs points are covered by the new edge
    #         elements_to_remove = []
    #         for _lfb in self.LFBs:

    #             overlaps = left_point[0] < _lfb[0] and high_right_point[0] > _lfb[0]
    #             left_edge_equal = left_point[0] == _lfb[0] and high_right_point[0] >= _lfb[0]
    #             right_edge_equal = (left_point[0] < _lfb[0] and high_right_point[0] >= _lfb[0])
    #             if (  # covering condition
    #                     overlaps or left_edge_equal or right_edge_equal
    #             ):
    #                 elements_to_remove.append(_lfb)
    #             print(overlaps, left_edge_equal, right_edge_equal, _lfb)

    #             # the following removes parts on flat
    #             # lfb is neighbor on right; we need to remove high_right_point
    #             if high_right_point[1] == _lfb[1] and high_right_point[0] == _lfb[0]:
    #                 if high_right_point in lfbs_to_add:
    #                     lfbs_to_add.remove(high_right_point)
    #                     elements_to_remove(high_right_point)

    #             # lfb is neighbor on left
    #             if high_right_point[1] == _lfb[1] and left_point[0] == _lfb[0]:
    #                 if left_point in lfbs_to_add:
    #                     lfbs_to_add.remove(left_point)
    #                     elements_to_remove(left_point)
    #             
    #         for element in elements_to_remove:
    #             if element in self.LFBs:
    #                 self.LFBs.remove(element)

    #         for element in lfbs_to_add:
    #             self.LFBs.add(element)

    #         bins_processed.append(_bin)
    #         DataGenerator().visualize_2D(bins_processed, self.w, self.rows, extreme_points=self.LFBs)

    #     return reward

    def is_bin_outside_borders(self, _bin):
        position_to_put_bin_into = self.LFBs[0]
        left_border = self.LFBs[0][0]

        closest_right = self._get_closest_right()

        ret = False
        if left_border +  _bin[0] > closest_right:  # clashes with box on right or total box border
            ret = True

        if self.LFBs[0][1] + _bin[1] > self.rows:  # taller than total box
            ret = True

        if ret:
            print(f'bin {_bin} could not fit into {self.LFBs} (closest_right: {closest_right}')

        return ret


    def _get_closest_right_point(self):
        left_border = self.LFBs[0][0]
        right_border = None

        closest_right = sorted((x for x in self.LFBs if x[0] > left_border), key=lambda x: x[1])
        if not closest_right:
            closest_right = (self.cols, 0)
        else:
            closest_right = closest_right[0]
        return closest_right

    def _get_closest_right(self):
        return self._get_closest_right_point()[0]


    def visualize_grid(self):

        import matplotlib
        matplotlib.use('GTK')
        np.random.seed(4)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.set_xticks(list(range(self.cols)))
        ax1.set_yticks(list(range(self.rows)))
        ax1.imshow(self.grid)
        plt.xlim(0, self.cols)
        plt.ylim(0, self.rows)
        plt.figure(1)
        plt.show()
        #plt.pause(0.2)
        plt.close()

    @staticmethod
    def get_next_turn(state, tile, val=1, get_only_success=False, destroy_state=False, colorful_states=False):
        '''
        destroy_state - will ruin the state.board, populating it with val
        colorful_state - not all occupied states are 1; for visualization purposes
        '''
        next_position = SolutionChecker.get_next_lfb_on_grid(state.board)
        # one without the other should not be possible
        if not next_position and len(state.tiles) == 0:
            print('solution found!')
            return ALL_TILES_USED, None
        elif not next_position:
            return NO_NEXT_POSITION_TILES_UNUSED, None

        if destroy_state:
            board = state.board
        else:
            board = np.copy(state.board)
        success, new_board = SolutionChecker.place_element_on_grid_given_grid(
            tile, next_position,
            val, board, get_cols(state.board), get_rows(state.board),
            get_only_success=get_only_success, colorful_states=colorful_states)

        if not success:
            # cannot place the tile. this branch will not be considered
            return TILE_CANNOT_BE_PLACED, None
        return True, new_board


    @staticmethod
    def get_next_occupied_col(board, next_position, colorful_states=False):
        that_position_row_until_next_occupied = board[
            next_position[0], next_position[1]:]

        if colorful_states:
            cols_with_non_zero = np.where(that_position_row_until_next_occupied!=0)
            if cols_with_non_zero[0].size == 0:
                return board.shape[1]
            res_col = next_position[1] + cols_with_non_zero[0][0]
        else:
            res = search.find_first(GLOBAL_OCCUPIED_VAL, that_position_row_until_next_occupied)
            if res == -1:
                return board.shape[1]
            res_col = next_position[1] + res 
        return res_col

    @staticmethod
    def get_valid_next_moves(state, tiles, val=1, colorful_states=False):
        possible_tile_moves = []
        next_lfb = SolutionChecker.get_next_lfb_on_grid(state.board)
        next_occupied_col = SolutionChecker.get_next_occupied_col(state.board, next_lfb, colorful_states=colorful_states)
        max_allowed_col_size = next_occupied_col - next_lfb[1] 
        max_height = state.board.shape[0]
        for tile in tiles:
            if tile[1] <= max_allowed_col_size and tile[0] + next_lfb[0] <= max_height:
                possible_tile_moves.append(tile)
        return possible_tile_moves

    @staticmethod
    def eliminate_pair_tiles(tiles, tile_to_remove):
        '''
        search through the list to find rotated instance, 
        then remove both
        '''
        index = tiles.index(tile_to_remove)
        new_tiles = tiles[:index] + tiles[index + 1:]

        rotated_tile = (tile_to_remove[1], tile_to_remove[0])
        rotated_tile_index = new_tiles.index(rotated_tile)

        new_tiles = new_tiles[:rotated_tile_index] + new_tiles[rotated_tile_index + 1:]
        return new_tiles


    @staticmethod
    def get_possible_tile_actions_given_grid(grid, tiles, pad_with_zeros=False):
        '''
        given a grid and tiles return the tiles which can be placed in lfb
        '''
        next_lfb = SolutionChecker.get_next_lfb_on_grid(grid)

        orig_tiles_length = len(tiles)
        new_tiles = []
        rows, cols = grid.shape
        for i, tile in enumerate(tiles):
            success, _ = SolutionChecker.place_element_on_grid_given_grid(
                tile, next_lfb,
                val=1, grid=grid, cols=cols, rows=rows, get_only_success=True
            )
            if not success:
                continue
            new_tiles.append(tile)

        if pad_with_zeros:
            new_tiles = SolutionChecker.pad_tiles_with_zero_scalars(
                new_tiles, orig_tiles_length - len(new_tiles))
        return new_tiles

    @staticmethod
    def get_valid_tile_actions_indexes_given_grid(grid, tiles):
        '''
        returns indexes of actions which can be performed
        '''
        next_lfb = SolutionChecker.get_next_lfb_on_grid(grid)

        tiles_indexes = []
        rows, cols = grid.shape
        for i, tile in enumerate(tiles):
            if tile[0] == 0 and tile[1] == 0:
                tiles_indexes.append(0)
                continue

            success, _ = SolutionChecker.place_element_on_grid_given_grid(
                tile, next_lfb,
                val=1, grid=grid, cols=cols, rows=rows, get_only_success=True
            )
            if not success:
                tiles_indexes.append(0)
            else:
                tiles_indexes.append(1)
        return tiles_indexes

    @staticmethod
    def pad_tiles_with_zero_scalars(tiles, n_zero_tiles_to_add):
        '''
        add tiles with zero matrices to compensate for tiles which were already placed
        '''
        new_tiles = tiles[:]
        for i in range(n_zero_tiles_to_add):
            new_tiles.append([0, 0])
        return new_tiles

    @staticmethod
    def tiles_to_np_array(tiles):
        return np.array([np.array(x) for x in tiles])

    @staticmethod
    def np_array_to_tiles(tiles):
        return [list(x) for x in tiles]

    @staticmethod
    def get_n_nonplaced_tiles(_tiles_ints):
        _tiles_ints = SolutionChecker.np_array_to_tiles(_tiles_ints)
        n_possible_tiles = len([x for x in _tiles_ints if x != [0, 0]])
        return n_possible_tiles

    @staticmethod
    def is_any_moves_left(tiles):
        return get_n_possible_tiles(tiles) != 0

    @staticmethod
    def get_tiles_with_orientation(tiles):
        if isinstance(tiles, np.ndarray): 
            tiles = tiles.tolist()
        tiles_with_orientations = tiles[:]
        for tile in tiles:
            tiles_with_orientations.append((tile[1], tile[0]))
        return tiles_with_orientations


def get_cols(board):
    return board.shape[1]

def get_rows(board):
    return board.shape[0]

