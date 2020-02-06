from collections import namedtuple
import numpy as np

from data_generator import DataGenerator

DEFAULT_HEIGHT = 8
DEFAULT_WIDTH = 8
ORIENTATIONS = 2

WinState = namedtuple('WinState', 'is_ended winner')


class Board():
    """
    BinPack Board.
    """

    def __init__(self, height, width, tiles, state=None, visualization_state=None):
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


        if state is None:
            board = np.zeros([1, self.height, self.width])
            self.state = np.concatenate((board, self.tiles), axis=0)
            self.vis_state = np.zeros([self.height, self.width])
        else:
            self.state = state
            self.vis_state = visualization_state
        assert self.state.shape == (len(tiles) + 1, self.height, self.width)

    def add_tile(self, position, player):
        """
        Create copy of board containing new tile.
        Position is index (?) on which to place tile.
        We always place the tile which is located at position 1 or 2. 
        """
        new_stack, vis_state = DataGenerator.play_position(
            self.state, position, tile_index=DataGenerator.get_n_tiles_placed(self.state),
            vis_state=self.vis_state
        )
        self.state = new_stack
        self.vis_state = vis_state
        return new_stack, vis_state


    def get_valid_moves(self):
        """
        Any drop on locus with zero value is a valid move
        If lower than self.height * self.width it is first orientation, 
        if not it is second
        """
        tiles = [self.state[1], self.state[2]]
        final_mask = DataGenerator.get_valid_moves_mask(self.state[0], tiles)
        final_mask = np.reshape(
            final_mask, (final_mask.shape[0] * final_mask.shape[1]))
        return final_mask

    def all_tiles_placed(self):
        ret = (DataGenerator.get_n_tiles_placed(self.state) ==
               (len(self.state) - 1) // self.orientations)
        return ret

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
        state = np.copy(state)
        return Board(self.height, self.width, self.tiles, state, vis_state)

    def __str__(self):
        result_str = ''
        for _slice in self.state:
            for row in _slice:
                result_str += str(row)
        return result_str
