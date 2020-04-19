from collections import namedtuple
import numpy as np

from data_generator import DataGenerator
from solution_checker import SolutionChecker

DEFAULT_HEIGHT = 8
DEFAULT_WIDTH = 8
ORIENTATIONS = 2

WinState = namedtuple('WinState', 'is_ended winner')


class Board():
    """
    BinPack Board.
    """
    ORIENTATIONS = 2

    def __init__(self, height, width, tiles, n_tiles, state=None, visualization_state=None):
        "Set up initial board configuration."
        """
        tiles - array of tiles with (width, height)
        state - current bin configuration with 1 indicating locus is taken
                    0 indicating that it is free
        """

        self.height = height
        self.width = width
        self.tiles = tiles
        self.orientations = ORIENTATIONS
        self.n_tiles = n_tiles


        if state is None:
            board = np.zeros([self.height, self.width])
            self.state = (board, self.tiles)
            self.vis_state = np.zeros([self.height, self.width])
        else:
            self.state = (state, self.tiles)
            self.vis_state = visualization_state

        assert len(self.state) == 2
        assert self.state[0].shape == (self.height, self.width)

    def add_tile(self, position, player):
        """
        Create copy of board containing new tile.
        Position is index (?) on which to place tile.
        We always place the tile which is located at position 1 or 2. 
        """

        next_lfb = SolutionChecker.get_next_lfb_on_grid(self.state[0])
        success, grid = SolutionChecker.place_element_on_grid_given_grid(
            self.tiles[position],
            next_lfb, val=1, grid=self.state[0], cols=self.width, rows=self.height
        )
        if success:
            tiles = [tuple(x) for x in self.tiles]
            tiles = SolutionChecker.eliminate_pair_tiles(tiles, tuple(self.tiles[position]))
            zero_tiles_to_add = self.ORIENTATIONS * self.n_tiles - len(tiles)
            self.tiles = SolutionChecker.pad_tiles_with_zero_scalars(
                tiles, zero_tiles_to_add)

        self.state = (grid, self.tiles)

        #TODO: what is vis_state
        self.vis_state = self.state[0]
        return self.state, self.vis_state


    def get_valid_moves(self):
        """
        Any drop on locus with zero value is a valid move
        If lower than self.height * self.width it is first orientation, 
        if not it is second
        """
        _tiles_ints = SolutionChecker.get_valid_tile_actions_indexes_given_grid(
            self.state[0], self.state[1])
        # _tiles_ints = tiles_to_np_array(SolutionChecker.pad_tiles_with_zero_scalars(_tiles_ints, ORIENTATIONS * n_tiles - len(_tiles_ints)))

        return np.array(_tiles_ints)

    def all_tiles_placed(self):
        for tile in self.tiles:
            if not np.array_equal(tile, [0, 0]):
                return False
        return True

    def get_win_state(self):
        if self.all_tiles_placed() or not np.any(self.get_valid_moves()): 
            # game  has ended calculate reward
            locus_filled = np.sum(self.state[0])
            total_locus = self.state[0].shape[0] * self.state[0].shape[1]
            if locus_filled == total_locus:
                return 1
            else:
                return locus_filled / total_locus

        # game has not ended yet
        return False


    def with_state(self, state, vis_state=None):
        """Create copy of board with specified pieces."""
        if state is None:
            state = self.state
        state_board = np.copy(state[0])
        state_tiles = np.copy(state[1])
        return Board(self.height, self.width, state_tiles, self.n_tiles, state_board, vis_state)

    def __str__(self):
        result_str = ''
        for _slice in self.state:
            for row in _slice:
                result_str += str(row)
        return result_str
