import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .BinPackLogic import Board

from data_generator import DataGenerator

class BinPackGame(Game):
    """
    BinPack Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=None, width=None, n_tiles=None):
        self.initialize(height, width, n_tiles)

    def initialize(self, height, width, n_tiles):
        Game.__init__(self)

        self.height = height
        self.width = width
        self.n_tiles = n_tiles

        dg = DataGenerator()
        #self.tiles = dg.gen_matrix_instance(n_tiles, width, height)
        # print(self.tiles)

        #self._base_board = Board(height, width, self.tiles)

    def getInitBoard(self):
        # self.initialize(self.height, self.width, self.n_tiles)
        return self._base_board.state, self._base_board.vis_state

    def getBoardSize(self):
        ORIENTATIONS = 2
        return (
            self.height, self.width,
            self.n_tiles * ORIENTATIONS + 1
                )

    def getActionSize(self):
        # allowing moves without gravity physical support
        return (self.width * self.height)
        # orientations are included in tiles
        # self._base_board.orientations)

    def getNextState(self, board, player, action, vis_state=None):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_state(state=np.copy(board), vis_state=vis_state)
        b.add_tile(action, player)
        return b.state, player, b.vis_state

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        return self._base_board.with_state(state=board).get_valid_moves()

    def getGameEnded(self, board, player):
        b = self._base_board.with_state(state=board)
        winstate = b.get_win_state()
        if winstate is False:
            # 0 used to represent unfinished game.
            return 0
        else:
            return winstate

    def getCanonicalForm(self, state, player):
        # Flip player from 1 to -1
        return state

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        """
        TODO:
        We need to rotate just the board and keep the pieces unrotated
        although I don't think rotating te pieces would hurt, it's just not  needed
        """
        # return [(board, pi), (board[:, ::-1], pi[::-1])]
        board_right = np.copy(board)
        board_right[0] = board_right[0][::-1]
        # return [(board, pi), (board_right, pi[::-1])]
        return [(board, pi)]

    def stringRepresentation(self, board):
        return str(self._base_board.with_state(state=board)), self._base_board.with_state(state=board)


def display(board):
    print(" -----------------------")
    print(' '.join(map(str, range(len(board[0])))))
    print(board)
    print(" -----------------------")
