import unittest
import numpy as np
from data_generator import DataGenerator
from solution_checker import SolutionChecker

from sortedcontainers import SortedKeyList
from state import State

class TestDataGenerator(unittest.TestCase):

    def test_gen_instance_visual_22(self):

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


    def test_check_perfect_solution(self):
        n = 20
        w = 40
        h = 40
        dg = DataGenerator()
        some_instance_visual = dg.gen_instance_visual(n, w, h)
        perfect_bin_configuration = sorted(some_instance_visual, key=lambda x: (x[2][0], x[2][1]))
        some_instance_np_array = dg._transform_instance_visual_to_np_array(some_instance_visual)

        solution_checker = SolutionChecker(n, h, w)
        self.assertEqual(
            solution_checker.get_reward(np.array(perfect_bin_configuration)),
            0
        )

    def test_check_imperfect_solution(self):
        n = 4
        cols = 10
        rows = 5
        dg = DataGenerator()
        some_instance_np_array = np.array(
            [[1, 10, 1], [2, 10, 2], [1, 10, 3], [5, 10, 4], [1, 10, 1]])

        solution_checker = SolutionChecker(n, cols, rows)
        self.assertEqual(
            solution_checker.get_reward(some_instance_np_array),
            (10 * 5) / (cols * rows)
        )

    def test_check_imperfect_solution_2(self):
        n = 4
        cols = 10
        rows = 5
        dg = DataGenerator()
        some_instance_np_array = np.array(
            [[1, 10, 1], [6, 10, 2], [2, 10, 3], [1, 10, 4], [1, 10, 1]])

        solution_checker = SolutionChecker(n, cols, rows)
        self.assertEqual(
            solution_checker.get_reward(some_instance_np_array),
            (10 * 6) / (cols * rows)
        )
        
    def test_check_imperfect_solution_count_files_2(self):
        n = 4
        cols = 10
        rows = 5
        dg = DataGenerator()
        some_instance_np_array = np.array(
            [[1, 10, 1], [6, 10, 2], [2, 10, 3], [1, 10, 4], [1, 10, 1]])

        solution_checker = SolutionChecker(n, cols, rows)
        self.assertEqual(
            solution_checker.get_reward(some_instance_np_array, count_tiles=True),
            1 / n
        )

    def test_check_imperfect_solution_count_tiles(self):
        n = 4
        cols = 10
        rows = 5
        dg = DataGenerator()
        some_instance_visual = dg.gen_instance_visual(n, cols, rows)
        # NOTE: first bin always repeated
        some_instance_np_array = np.array(
            [[1, 10, 1], [2, 10, 2], [1, 10, 3], [5, 10, 4], [1, 10, 1]])

        solution_checker = SolutionChecker(n, cols, rows)
        self.assertEqual(
            solution_checker.get_reward(some_instance_np_array, count_tiles=True),
            1 / n
        )

    def test_bin_outside_border(self):

        n = 20
        h = 50
        w = 50

        solution_checker = SolutionChecker(n, h, w)
        #
        # 11  -------------
        #     |           |
        #     |           |          |
        #     |           |          |
        #     -----------------------|
        #                40

        solution_checker.LFBs = SortedKeyList([], key=lambda x: (x[1], x[0]))
        solution_checker.LFBs.add((40, 11))

        _bin = (10, 10)
        self.assertFalse(solution_checker.is_bin_outside_borders(_bin))

        _bin = (12, 10)
        self.assertTrue(solution_checker.is_bin_outside_borders(_bin))

    def test_get_next_lfb_on_grid(self):
        state = np.array([[1, 1], [0, 0]])
        res = SolutionChecker.get_next_lfb_on_grid(state)
        self.assertEqual(res, (1, 0)) 

    def test_get_next_lfb_on_grid_2(self):
        state = np.array([[1, 1, 0], [0, 0, 0]])
        res = SolutionChecker.get_next_lfb_on_grid(state)
        self.assertEqual(res, (0, 2)) 

    def test_get_next_lfb_on_grid_3(self):
        state = np.array([[1, 1, 1], [1, 1, 1]])
        res = SolutionChecker.get_next_lfb_on_grid(state)
        self.assertIsNone(res) 

    def test_get_next_lfb_on_grid_4(self):
        state = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        res = SolutionChecker.get_next_lfb_on_grid(state)
        self.assertEqual(res, (2, 0)) 

    def test_get_next_lfb_on_grid_5(self):
        state = np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 1]
            ])
        res = SolutionChecker.get_next_lfb_on_grid(state)
        self.assertEqual(res, (2, 0)) 

class TestNextTurn(unittest.TestCase):

    def setUp(self):
        self.board = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        self.tiles = [(2, 1), (1, 2)]
        self.tile = [2, 1]
        self.state = State(self.board, self.tiles)

    def test_get_next_turn(self):

        success, new_board,  = SolutionChecker.get_next_turn(
            self.state, self.tile, val=1, get_only_success=False, destroy_state=False
        )

        self.assertEqual(success, True)
        np.testing.assert_array_equal(
            new_board,
            np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 1]
            ])
        )
        # assert board is not destroyed
        np.testing.assert_array_equal(
            self.state.board,
            np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [0, 0, 0, 0]
            ])
        )

    def test_get_next_turn_destroy_board(self):

        success, new_board,  = SolutionChecker.get_next_turn(
            self.state, self.tile, val=1, get_only_success=False, destroy_state=True
        )
        self.assertEqual(success, True)

        # assert board is destroyed i.e. equal to new_board
        np.testing.assert_array_equal(
            self.state.board,
            new_board
        )

    def test_get_next_turn_only_success(self):

        success, new_board,  = SolutionChecker.get_next_turn(
            self.state, self.tile, val=1, get_only_success=True, destroy_state=True
        )

        # assert board is destroyed i.e. equal to new_board
        self.assertIsNone(new_board)

class TestGetValidNextMoves(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_valid_next_moves_1(self):
        self.board = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        self.tiles = [(2, 1), (1, 2)]
        self.state = State(self.board, self.tiles)
        next_moves = SolutionChecker.get_valid_next_moves(self.state, self.tiles)
        self.assertEqual(next_moves, [(2, 1)])

    def test_get_valid_next_moves_2(self):
        self.board = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        tiles_list = [(2, 1), (1, 2), (3, 3), (4, 4), (1, 1)]

        self.tiles = sorted(tiles_list, key=lambda x: (x[1], x[0]))
        self.state = State(self.board, self.tiles)
        next_moves = SolutionChecker.get_valid_next_moves(self.state, self.tiles)
        self.assertEqual(next_moves, [(1, 1), (2, 1)])

    def test_get_valid_next_moves_3(self):
        self.board = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1]
        ])
        tiles_list = [(2, 1), (1, 2), (3, 3), (3, 2), (1, 1), (2, 3)]

        self.tiles = sorted(tiles_list, key=lambda x: (x[1], x[0]))
        self.state = State(self.board, self.tiles)
        next_moves = SolutionChecker.get_valid_next_moves(self.state, self.tiles)
        self.assertEqual(next_moves, [(1, 1), (2, 1), (1, 2), (3, 2)])

    def test_get_valid_next_moves_3(self):
        self.board = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1]
        ])
        tiles_list = [(2, 1), (1, 2), (3, 3), (3, 2), (1, 1), (6, 1)]

        self.tiles = sorted(tiles_list, key=lambda x: (x[1], x[0]))
        self.state = State(self.board, self.tiles)
        next_moves = SolutionChecker.get_valid_next_moves(self.state, self.tiles)
        self.assertEqual(next_moves, [(1, 1), (2, 1), (1, 2), (3, 2)])

    def test_get_valid_next_moves_4(self):
        self.board = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0]
        ])
        tiles_list = [(2, 1), (1, 2), (3, 3), (3, 2), (1, 1), (6, 1), (3, 1)]

        self.tiles = sorted(tiles_list, key=lambda x: (x[1], x[0]))
        self.state = State(self.board, self.tiles)
        next_moves = SolutionChecker.get_valid_next_moves(self.state, self.tiles)
        self.assertEqual(next_moves, [(1, 1), (2, 1), (3, 1)])

    def test_get_valid_next_moves_5(self):
        self.board = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ])
        tiles_list = [(2, 3), (2, 4), (3, 3)]

        self.tiles = sorted(tiles_list, key=lambda x: (x[1], x[0]))
        self.state = State(self.board, self.tiles)
        next_moves = SolutionChecker.get_valid_next_moves(self.state, self.tiles)
        self.assertEqual(next_moves, [(2, 3), (3, 3)])


class TestEliminatePairTiles(unittest.TestCase):

    def setUp(self):
        pass

    def test_eliminate_pair_tiles(self):
        tiles_list = [(1, 1), (2, 1), (1, 1), (1, 2)]

        new_tiles = SolutionChecker.eliminate_pair_tiles(tiles_list, tile_to_remove=(1, 1))
        self.assertEqual(new_tiles, [(2, 1), (1, 2)])

    def test_eliminate_pair_tiles_2(self):
        tiles_list = [(8, 1), (2, 1), (1, 1), (1, 8)]

        new_tiles = SolutionChecker.eliminate_pair_tiles(tiles_list, tile_to_remove=(1, 8))
        self.assertEqual(new_tiles, [(2, 1), (1, 1)])

    def test_eliminate_pair_tiles_3(self):
        tiles_list = [(1, 1), (2, 1), (1, 1), (1, 1), (2, 1), (1, 1)]

        new_tiles = SolutionChecker.eliminate_pair_tiles(tiles_list, tile_to_remove=(1, 1))
        self.assertEqual(new_tiles, [(2, 1), (1, 1), (2, 1), (1, 1)])

    def test_eliminate_pair_tiles_when_pair_is_not_present(self):
        tiles_list = [(1, 8), (2, 1), (1, 1), (1, 1), (2, 1), (1, 1)]

        with self.assertRaises(ValueError):
            SolutionChecker.eliminate_pair_tiles(tiles_list, tile_to_remove=(1, 8))


class TestGetPossibleTileActions(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_possible_tile_actions_given_grid(self):
        grid = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ])
        tiles = [[1, 2], [2, 1], [4, 5], [1, 1], [1, 1], [3, 2], [3, 3], [2, 3]]
        possible_tiles_to_place = SolutionChecker.get_possible_tile_actions_given_grid(grid, tiles)

        self.assertEqual(possible_tiles_to_place, [[1, 2], [2, 1], [1, 1], [1, 1], [3, 2]])

class TestPadTilesWithZeroScalars(unittest.TestCase):
    def setUp(self):
        pass

    def test_pad(self):
        tiles = [[1, 2], [2, 1], [4, 5]]
        padded_tiles = SolutionChecker.pad_tiles_with_zero_scalars(tiles, 2)

        self.assertEqual(len(padded_tiles), 5)
        self.assertEqual([[1, 2], [2, 1], [4, 5], [0, 0], [0, 0]], padded_tiles)

    def test_get_valid_actions_indexed_given_grid(self):

        grid = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ])
        tiles = [[1, 2], [2, 1], [0, 0], [4, 5], [1, 1], [1, 1], [3, 2], [3, 3], [2, 3]]
        possible_tiles_indexes = SolutionChecker.get_valid_tile_actions_indexes_given_grid(
            grid, tiles)

        self.assertEqual(possible_tiles_indexes, [1, 1, 0, 0, 1, 1, 1, 0, 0])

if __name__=='__main__':
    unittest.main()
