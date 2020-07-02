import matplotlib
from itertools import cycle
import csv
import random
import numpy as np
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from collections import defaultdict, namedtuple, OrderedDict

InstanceFromFile = namedtuple('InstanceFromFile', ['bins', 'id', 'n_tiles_placed'])

ORIENTATIONS = 2
class DataGenerator(object):

    def __init__(self, w=None, h=None):
        self.w = w
        self.h = h
        self.frozen_first_batch = None
        self.instances_from_file = defaultdict(lambda: OrderedDict())

    def gen_instance_visual(self, n, w, h, dimensions=2, seed=None): # Generate random bin-packing instance
        no_duplicates = False
        self.w = w
        self.h = h
        if seed is not None:
            np.random.seed(seed)

        bins = [[w, h, (0, 0)]]
        while len(bins) <  n:
            random_bin_index = random.randint(0, len(bins) - 1)
            bin_to_split = bins[random_bin_index]

            axis_to_split = np.random.randint(0, 2, size=1)[0]

            
            if bin_to_split[axis_to_split] <= 1:
                # cant split anymore; this is minimum size
                continue

            random_bin = bins[random_bin_index]

            split_val = int(np.random.randint(1, bin_to_split[axis_to_split], size=1)[0])
            new_bins = self._split_bin(bin_to_split, axis_to_split, split_val)

            if no_duplicates:
                _bins_sizes = [(x[:2]) for x in bins]
                rotated_bin_1 = [new_bins[0][:2][1], new_bins[0][:2][0]]
                rotated_bin_2 = [new_bins[1][:2][1], new_bins[1][:2][0]]
                if (new_bins[0][:2] in _bins_sizes or
                        new_bins[1][:2] in _bins_sizes or
                        rotated_bin_1 in _bins_sizes or
                        rotated_bin_2 in _bins_sizes or
                        new_bins[0][:2] == new_bins[1][:2] 
                        or rotated_bin_2 == new_bins[0][:2]
                        or rotated_bin_1 == new_bins[1][:2]):
                    continue

            bins.pop(random_bin_index)
            bins.insert(random_bin_index, new_bins[0])
            bins.insert(random_bin_index, new_bins[1])
        return bins

    def _transform_instance_visual_to_np_array(self, bins, dimensions=2):
        _bins = np.array([x[:dimensions] for x in bins])
        solution = sorted(bins, key=lambda x: (x[2][0], x[2][1]))
        return _bins, solution

    def gen_instance(self, n, w, h, dimensions=2, seed=0): # Generate random bin-packing instance
        bins, solution = self._transform_instance_visual_to_np_array(self.gen_instance_visual(n, w, h, seed=seed), dimensions=dimensions)
        return np.array(bins), solution

    def gen_tiles_and_board(self, n, w, h, dimensions=2, seed=0, order_tiles=False, from_file=False): # Generate random bin-packing instance
        '''
        generates tiles of both orientations
        '''
        if from_file:
            instance_visual = self.gen_instance_from_file(n, w, h)
        else:
            instance_visual = self.gen_instance_visual(n, w, h, seed=seed)

        return self.transform_instance_visual_to_tiles_and_board(w, h, instance_visual, dimensions=dimensions, order_tiles=order_tiles)

    def transform_instance_visual_to_tiles_and_board(self, w, h, instance_visual, dimensions=2, order_tiles=False):

        tiles, solution = self._transform_instance_visual_to_np_array(instance_visual, dimensions=dimensions)
        new_tiles = []
        for tile in tiles:
            # 2 orientations
            new_tiles.append(tuple(tile))
            new_tiles.append((tile[1], tile[0]))
        if order_tiles or True:
            new_tiles = sorted(new_tiles, key=lambda x: (x[1], x[0]))

        board = np.zeros((w, h))
        return new_tiles, board

    def gen_matrix_instance(self, n, w, h, dimensions=2, seed=0, with_solution=False):
        bins, solution = self.gen_instance(
            n, w, h, dimensions=dimensions, seed=seed)
        if with_solution:
            return self._transform_instance_to_matrix(bins), solution
        else:
            return self._transform_instance_to_matrix(bins)


    def gen_instance_from_file(self, n, w, h):
        instances = self.read_instances()
        try:
            instance = instances[n][w, h][0].bins
        except KeyError:
            print('no such instance could be found in the file')
        return instance

    def read_instances(self):
        with open('puzzles_n20_with_solutions_2.csv') as csvfile:
            csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                print(row)
                new_instance = DataGenerator.x_y_str_to_bins_format(row['tiles'])
                self.instances_from_file[int(row['num_tiles'])].setdefault(
                    (int(row['board_width']), int(row['board_height'])), []).append(
                    InstanceFromFile(bins=new_instance, id=row['id'], n_tiles_placed=row['tiles_placed']))
        return self.instances_from_file

    @staticmethod
    def x_y_str_to_bins_format(x_y_string):
        _list = eval(x_y_string)
        ret_list = []
        unknown_position = (0, 0)
        for el in _list:
            ret_list.append([el['X'], el['Y'], unknown_position])
        return ret_list

    @staticmethod
    def tile_to_matrix(tile, w, h):
        _slice = np.zeros([h, w])
        for i in range(tile[0]):
            for j in range(tile[1]):
                _slice[i][j] = 1
        return _slice



    def _transform_instance_to_matrix(self, tiles, only_one_orientation=False):
        """
        transforms list of bins:
        [[2, 3], [4, 5]]
        to stacks of bins with 2 orientations in left bottom corner
        of size (w, h):
        0000000000000
        0000000000000
        1100000000000
        1100000000000
        1100000000000
        """
        h = self.h
        w = self.w

        all_slices = None
        orientations = range(ORIENTATIONS)
        if only_one_orientation:
            orientations = [0]

        for tile in tiles:
            for orientation in orientations:
                if orientation == 0:
                    _slice = self.tile_to_matrix(tile, w, h)
                else:
                    _slice = self.tile_to_matrix((tile[1], tile[0]), w, h)
                if all_slices is not None:
                    all_slices = np.dstack((all_slices, _slice))
                else:
                    all_slices = _slice
                    all_slices = np.reshape(all_slices, (_slice.shape[0], _slice.shape[1], 1))

        return all_slices


    @staticmethod
    def get_matrix_tile_dims(tile):
        #TODO: optimize
        matrix_rows, matrix_cols = tile.shape
        rows = 0
        cols = 0
        while tile[0][cols] == 1: 
            cols += 1
            if cols >= matrix_cols:
                break

        while tile[rows][0] == 1: 
            rows += 1
            if rows >= matrix_rows:
                break

        return (rows, cols)

    @staticmethod
    def get_valid_moves_mask(state, tiles):
        """
        state - 2d matrix representing current tile state
        tiles - list of 2 matrices with same size as state each presenting one orientation
        """
        rows = state.shape[0]
        cols = state.shape[1]

        for i, tile in enumerate(tiles):
            mask = state == 0
            for row in range(rows):
                for col in range(cols):
                    # no need to check this one as position is already taken
                    if mask[row][col] == False: 
                        continue

                    # checks if it clashes with already existing tiles
                    try:
                        DataGenerator.add_tile_to_state(
                            state, tile, (row, col))
                    except ValueError:
                        mask[row][col] = False

            if i == 0:
                first_mask = np.copy(mask)
            else:
                second_mask = np.copy(mask)

        final_mask = np.concatenate((first_mask, second_mask), axis=0)
        return final_mask

    @staticmethod
    def position_index_to_row_col(position, cols, rows):
        return (position // cols, position % cols)


    @staticmethod
    def play_position(stack, position, tile_index=None, vis_state=None):
        """
        Given a stack and position, add a tile to stack[0],
        and make tile and its rotation zeros,
        and move them to back of stack bringing forward new ones
        """
        new_stack = np.copy(stack)
        rows = new_stack.shape[0] 
        cols = new_stack.shape[1] 

        if position >= cols * rows:
            tile = new_stack[:, :, 2]
            position = position - rows * cols
        else:
            tile = new_stack[:, :, 1]

        new_stack = np.delete(new_stack, 1, axis=2)
        new_stack = np.delete(new_stack, 1, axis=2)

        new_stack = np.dstack((new_stack, np.zeros([rows, cols])))
        new_stack = np.dstack((new_stack, np.zeros([rows, cols])))

        position = DataGenerator.position_index_to_row_col(
            position, tile.shape[1], tile.shape[0]
        )

        ret = DataGenerator.add_tile_to_state(
            new_stack[:, :, 0], tile, position, tile_index=tile_index, vis_state=vis_state)

        new_stack[:, :, 0] = ret[0]
        return new_stack, ret[1]

    @staticmethod
    def get_n_tiles_placed(stack):
        """
        count how many tiles are placed
        assumes that all placed tiles are matrices with all zeros
        """
        count = 0
        for i in range(1, stack.shape[2]):
            _slice = stack[:, :, i]
            if not _slice.any():
                count += 1
        ORIENTATIONS = 2
        count = count / ORIENTATIONS
        return count

    @staticmethod
    def add_tile_to_state(state, tile, position, tile_index=0, vis_state=None):
        new_state = np.copy(state)
        if vis_state is not None:
            new_vis_state = np.copy(vis_state)
        else:
            new_vis_state = None
        tile_rows, tile_cols = DataGenerator.get_matrix_tile_dims(tile)
        for row in range(tile_rows):
            for col in range(tile_cols):
                if position[0] + row >= state.shape[0]:
                    raise ValueError(
                        f'tile goes out of bin height {position}')
                if position[1] + col >= state.shape[1]:
                    raise ValueError(
                        f'tile goes out of bin width {position}')

                if new_state[position[0] + row ][position[1] + col] == 1:
                    raise ValueError(
                        f'locus already taken:\n'
                        f'state:\n {state}\n tile:\n {tile}\n'
                        f'position: {position}'

                    )
                else:
                    new_state[position[0] + row ][position[1] + col] = 1
                    if vis_state is not None:
                        new_vis_state[position[0] + row ][position[1] + col] = tile_index + 1

        return new_state, new_vis_state

    def _split_bin(self, _bin, axis, value):
        assert len(_bin) == 3
        assert type(_bin[0]) == int, type(_bin[0])
        assert type(_bin[1]) == int, type(_bin[1])
        assert type(_bin[2]) == tuple, type(_bin[2])

        if axis == 0:
            ret = [ [value, _bin[1], _bin[2]], [_bin[0] - value, _bin[1], (_bin[2][0] + value, _bin[2][1])] ]
        elif axis == 1:
            ret = [ [_bin[0], value, _bin[2]], [_bin[0], _bin[1] - value, (_bin[2][0], _bin[2][1] + value)] ] 
        return ret

    def train_batch(self, batch_size, n, w, h, dimensions=2, seed=0, freeze_first_batch=False):
        input_batch = []
        if freeze_first_batch and self.frozen_first_batch:
            return self.frozen_first_batch
        for _ in range(batch_size):
            input_, solution = self.gen_instance(n, w, h, dimensions=dimensions, seed=seed)
            input_batch.append(input_)

        if freeze_first_batch:
            if not self.frozen_first_batch:
                self.frozen_first_batch = input_batch
                print('Using frozen batch', self.frozen_first_batch)
        return input_batch


    def test_batch(self, batch_size, n, w, h, dimensions=2, seed=0, shuffle=False): # Generate random batch for testing procedure
        input_batch = []
        input_, solution = self.gen_instance(n, w, h, dimensions=dimensions, seed=seed)
        for _ in range(batch_size): 
            sequence = np.copy(input_)
            if shuffle==True: 
                np.random.shuffle(sequence) # Shuffle sequence
            input_batch.append(sequence) # Store batch
        return input_batch

    def get_cmap(self, n, name='Accent'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name)

    def visualize_2D(self, bins, w, h, extreme_points=None): # Plot tour

        matplotlib.use('GTK')
        np.random.seed(4)
        cycol = cycle('bgrcmk')

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        cmap = self.get_cmap(len(bins))
        for i, _bin in enumerate(bins):
            color = np.random.rand(3,)
            ax1.add_patch(
                patches.Rectangle(
                    (_bin[2][0], _bin[2][1]), _bin[0], _bin[1],
                    # color=cmap(i),
                    color=color,
                    edgecolor=color,
                    hatch=patterns[random.randint(0, len(patterns) - 1)])
            )
            ax1.text(_bin[2][0] + _bin[0] / 2 - 2 , _bin[2][1] + _bin[1] / 2, str(_bin))

        if extreme_points:
            x = [x[0] for x in extreme_points]
            y = [x[1] for x in extreme_points]
            plt.scatter(x, y, s=500)

        ax1.set_xticks(list(range(w)))
        ax1.set_yticks(list(range(h)))
        ax1.grid(which='both')

        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.figure(1)
        plt.show()
        #plt.pause(0.2)
        #plt.close()
