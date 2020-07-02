import unittest
import numpy as np
from data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):

    def test_gen_instance_visual(self):

        n = 20
        w = 40
        h = 40
        dg = DataGenerator()
        some_instance = dg.gen_instance_visual(n, w, h)
        self.assertEqual(len(some_instance), n)
        for _bin in some_instance:
            self.assertEqual(len(_bin), 3)
            self.assertTrue(_bin[0] <= w)
            self.assertTrue(_bin[1] <= h)

        # dg.visualize_2D(some_instance, w, h)

    def test_gen_instance(self):

        n = 20
        w = 40
        h = 40
        dg = DataGenerator()
        some_instance, solution = dg.gen_instance(n, w, h)
        self.assertEqual(len(some_instance), n)
        self.assertEqual(len(solution), n)
        for _bin in some_instance:
            self.assertEqual(len(_bin), 2)
            self.assertTrue(_bin[0] <= w)
            self.assertTrue(_bin[1] <= h)
            self.assertEqual(type(some_instance), np.ndarray)

    def test_transform_instance_to_matrix(self):

        w = 40
        h = 30
        dg = DataGenerator(w, h)
        tiles = np.array([[2, 3], [4, 5]])
        matrix = dg._transform_instance_to_matrix(tiles)

        self.assertEqual(matrix[:,:,0].shape , (h, w))
        ORIENTATIONS = 2
        self.assertEqual(matrix.shape, (h, w, len(tiles) * ORIENTATIONS,))

        # first tile different orientation
        self.assertEqual(matrix[0][0][0] , 1)
        self.assertEqual(matrix[0][1][0] , 1)
        self.assertEqual(matrix[0][2][0] , 1)
        self.assertEqual(matrix[1][0][0] , 1)
        self.assertEqual(matrix[1][1][0] , 1)
        self.assertEqual(matrix[1][2][0] , 1)

        self.assertEqual(matrix[2][0][0] , 0)
        self.assertEqual(matrix[2][1][0] , 0)
        self.assertEqual(matrix[2][2][0] , 0)
        self.assertEqual(matrix[1][3][0] , 0)

        # first tile first orientation
        self.assertEqual(matrix[0][0][1] , 1)
        self.assertEqual(matrix[0][1][1] , 1)
        self.assertEqual(matrix[1][0][1] , 1)
        self.assertEqual(matrix[1][1][1] , 1)
        self.assertEqual(matrix[2][0][1] , 1)
        self.assertEqual(matrix[2][1][1] , 1)
                                        
        self.assertEqual(matrix[0][2][1] , 0)
        self.assertEqual(matrix[1][2][1] , 0)
        self.assertEqual(matrix[0][3][1] , 0)


    def test_get_matrix_tile_dims(self):
        w = 8
        h = 9
        dg = DataGenerator(w, h)
        tile_1_rows = 2
        tile_1_cols = 3


        tiles = np.array([[tile_1_rows, tile_1_cols], [4, 5]])
        matrix = dg._transform_instance_to_matrix(tiles)

        self.assertEqual(matrix[:, :, 0].shape , (h, w))
        ORIENTATIONS = 2
        self.assertEqual(matrix.shape, (h, w, len(tiles) * ORIENTATIONS,))

        self.assertEqual(dg.get_matrix_tile_dims(matrix[:, :, 0]), (tile_1_rows, tile_1_cols))
        self.assertEqual(dg.get_matrix_tile_dims(matrix[:, :, 1]), (tile_1_cols, tile_1_rows))
        self.assertEqual(dg.get_matrix_tile_dims(matrix[:, :, 2]), (4, 5))
        self.assertEqual(dg.get_matrix_tile_dims(matrix[:, :, 3]), (5, 4))

    def test_tile_to_matrix(self):
        tile = [2, 5]
        w = 8
        h = 9
        res = DataGenerator.tile_to_matrix(tile, w, h)

        self.assertEqual(list(res[0]), [1., 1., 1., 1., 1., 0., 0., 0.,])
        self.assertEqual(list(res[1]), [1., 1., 1., 1., 1., 0., 0., 0.,])
        self.assertEqual(list(res[2]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(res[3]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(res[4]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(res[5]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(res[6]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(res[7]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(res[8]), [0., 0., 0., 0., 0., 0., 0., 0.,])


    def test_add_tile_to_state(self):
        """
        test that tile gets added to matrix at depth[0]
        """
        w = 8
        h = 9
        dg = DataGenerator(w, h)
        tiles = np.array([[2, 3], [2, 1]])
        matrix = dg._transform_instance_to_matrix(tiles)

        self.assertEqual(list(matrix[:,:,0][0]), [1., 1., 1., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(matrix[:,:,0][1]), [1., 1., 1., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(matrix[:,:,0][2]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(matrix[:,:,0][3]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(matrix[:,:,0][4]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(matrix[:,:,0][5]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(matrix[:,:,0][6]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(matrix[:,:,0][7]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(matrix[:,:,0][8]), [0., 0., 0., 0., 0., 0., 0., 0.,])

        state = matrix[:,:,0].copy()
        state, vis = DataGenerator.add_tile_to_state(state, DataGenerator.tile_to_matrix([2, 1], w, h), (2, 2))

        self.assertEqual(list(state[0]), [1., 1., 1., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(state[1]), [1., 1., 1., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(state[2]), [0., 0., 1., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(state[3]), [0., 0., 1., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(state[4]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(state[5]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(state[6]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(state[7]), [0., 0., 0., 0., 0., 0., 0., 0.,])
        self.assertEqual(list(state[8]), [0., 0., 0., 0., 0., 0., 0., 0.,])

    def test_add_tile_out_of_bounds_raises_error(self):
        w = 40
        h = 30
        dg = DataGenerator(w, h)
        tiles = np.array([[2, 3], [4, 5]])
        matrix = dg._transform_instance_to_matrix(tiles)

        state = np.copy(matrix[0])

        self.assertRaises(
            ValueError,
            lambda: DataGenerator.add_tile_to_state(state, DataGenerator.tile_to_matrix([1, 3], w, h), (38, 2))
            )

    def test_add_tile_to_state_raises_error(self):
        w = 40
        h = 30
        dg = DataGenerator(w, h)
        tiles = np.array([[2, 3], [4, 5]])
        matrix = dg._transform_instance_to_matrix(tiles)

        state = np.copy(matrix[0])

        self.assertRaises(
            ValueError,
            lambda: DataGenerator.add_tile_to_state(state, DataGenerator.tile_to_matrix([4, 5], w, h), (1, 2)))

    def test_get_valid_moves_mask(self):
        w = 8
        h = 6
        dg = DataGenerator(w, h)

        #             rows, cols  rows, cols
        tiles = np.array([[2, 3], [4, 5]])
        matrix = dg._transform_instance_to_matrix(tiles)

        state = np.copy(matrix[:, :, 0])
        mask = DataGenerator.get_valid_moves_mask(state, [matrix[:, :, 2], matrix[:, :, 3]])
        """
        state is:
          [[1. 1. 1. 0. 0. 0. 0. 0.]
           [1. 1. 1. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0.]]
        """

        self.assertEqual(list(mask[0]), [False, False, False,  True, False, False, False, False])
        self.assertEqual(list(mask[1]), [False, False, False,  True, False, False, False, False])
        self.assertEqual(list(mask[2]), [True,   True,  True,  True, False, False, False, False])
        self.assertEqual(list(mask[3]), [False, False, False, False, False, False, False, False])
        self.assertEqual(list(mask[4]), [False, False, False, False, False, False, False, False])
        self.assertEqual(list(mask[5]), [False, False, False, False, False, False, False, False])

        self.assertEqual(list(mask[6] ), [False, False, False,  True,  True, False, False, False])
        self.assertEqual(list(mask[7] ), [False, False, False,  True,  True, False, False, False])
        self.assertEqual(list(mask[8] ), [False, False, False, False, False, False, False, False])
        self.assertEqual(list(mask[9] ), [False, False, False, False, False, False, False, False])
        self.assertEqual(list(mask[10]), [False, False, False, False, False, False, False, False])
        self.assertEqual(list(mask[11]), [False, False, False, False, False, False, False, False])


    def test_play_position(self):
        w = 8
        h = 8

        dg = DataGenerator(w, h)
        board = np.zeros([h, w, 1])
        tiles = np.array([[2, 3], [4, 5]])
        tiles = dg._transform_instance_to_matrix(tiles)
        stack = np.dstack((board, tiles))

        self.assertEqual(list(stack[:, :, 0][0]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 0][1]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 0][2]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 0][3]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 0][4]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 0][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 0][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 0][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(list(stack[:, :, 1][0]), [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 1][1]), [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 1][2]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 1][3]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 1][4]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 1][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 1][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 1][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(list(stack[:, :, 2][0]), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 2][1]), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 2][2]), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 2][3]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 2][4]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 2][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 2][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 2][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(list(stack[:, :, 3][0]), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 3][1]), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 3][2]), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 3][3]), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 3][4]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 3][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 3][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 3][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(list(stack[:, :, 4][0]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 4][1]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 4][2]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 4][3]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 4][4]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 4][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 4][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(stack[:, :, 4][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        print(stack.shape)
        new_stack, vis_new_stack = DataGenerator.play_position(stack, 64)

        self.assertEqual(list(new_stack[:, :, 0][0]), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 0][1]), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 0][2]), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 0][3]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 0][4]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 0][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 0][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 0][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(list(new_stack[:, :, 1][0]), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 1][1]), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 1][2]), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 1][3]), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 1][4]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 1][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 1][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 1][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(list(new_stack[:, :, 2][0]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 2][1]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 2][2]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 2][3]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 2][4]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 2][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 2][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 2][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(list(new_stack[:, :, 3][0]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 3][1]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 3][2]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 3][3]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 3][4]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 3][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 3][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 3][7]), [0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(list(new_stack[:, :, 4][0]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 4][1]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 4][2]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 4][3]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 4][4]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 4][5]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 4][6]), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(new_stack[:, :, 4][7]), [0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_n_tiles_placed(self):
        w = 8
        h = 8

        dg = DataGenerator(w, h)
        board = np.zeros([h, w, 1])
        tiles = np.array([[2, 3], [4, 5]])
        tiles = dg._transform_instance_to_matrix(tiles)
        stack = np.dstack((board, tiles))
        print(stack.shape)
        self.assertEqual(DataGenerator.get_n_tiles_placed(stack), 0)

        new_stack, vis = DataGenerator.play_position(stack, 64)

        self.assertEqual(DataGenerator.get_n_tiles_placed(new_stack), 1)
        new_stack, vis = DataGenerator.play_position(new_stack, 32)
        self.assertEqual(DataGenerator.get_n_tiles_placed(new_stack), 2)


    def test_position_index_to_row_col(self):

        """

          | 0   1   2   3
        --|---------------
        0 | 0   1   2   3
          |
        1 | 4   5   6   7   
          |
        2 | 8   9  10  11
          |
        3 |12  13  14  15

        """


        position_index = 7
        width = 4
        height = 3

        self.assertEqual(DataGenerator.position_index_to_row_col(position_index, width, height), (1, 3))

        position_index = 3
        width = 4
        height = 3

        self.assertEqual(DataGenerator.position_index_to_row_col(position_index, width, height), (0, 3))

        position_index = 0
        width = 4
        height = 3

        self.assertEqual(DataGenerator.position_index_to_row_col(position_index, width, height), (0, 0))

        position_index = 5
        width = 4
        height = 3

        self.assertEqual(DataGenerator.position_index_to_row_col(position_index, width, height), (1, 1))

        position_index = 4
        width = 4
        height = 3

        self.assertEqual(DataGenerator.position_index_to_row_col(position_index, width, height), (1, 0))

        position_index = 11
        width = 4
        height = 3

        self.assertEqual(DataGenerator.position_index_to_row_col(position_index, width, height), (2, 3))


    def test_split_bin(self):
        dg = DataGenerator()
        _bin = [21, 41, (0, 0)]
        self.assertEqual(
            dg._split_bin(_bin, 0, 14),
            [[14, 41, (0, 0)], [21 - 14, 41, (14, 0)]]
        )
        self.assertEqual(
            dg._split_bin(_bin, 1, 14),
            [[21, 14, (0, 0)], [21, 41 - 14, (0, 14)]]
        )

    def test_train_batch(self):
        dg = DataGenerator()
        batch_size = 10
        n = 20
        batch = dg.train_batch(10, n, 40, 40)
        self.assertEqual(len(batch), 10)
        self.assertEqual(len(batch[0]), n)

    def test_test_batch(self):
        dg = DataGenerator()
        batch_size = 10
        n = 20
        batch = dg.test_batch(10, n, 40, 40)
        self.assertEqual(len(batch), 10)
        self.assertEqual(len(batch[0]), n)

    # def test_read_instances_from_csv(self):
    #     instances = DataGenerator().read_instances()
    #     assert False

    def test_read_instances_string_from_csv(self):
        bins = DataGenerator.x_y_str_to_bins_format(
            '[{"X":6,"Y":5},{"X":6,"Y":3},{"X":6,"Y":1},{"X":5,"Y":2},{"X":3,"Y":2},{"X":2,"Y":1}]')

        self.assertEqual(len(bins), 6)
        self.assertEqual(bins[0], [6, 5, (0, 0)])
