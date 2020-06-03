import math
import json
import numpy as np
import random
# EPS = 1e-8
EPS = 0.1

from collections import OrderedDict

from data_generator import DataGenerator
from state import State
from solution_checker import SolutionChecker, ALL_TILES_USED, TILE_CANNOT_BE_PLACED, NO_NEXT_POSITION_TILES_UNUSED, get_cols, get_rows

ORIENTATIONS = 2

def sort_key(x):
    return x.score


def get_max_index(_list):
    max_index = 0
    max_value = -math.inf
    for i, el in enumerate(_list):
        if not el:
            continue
        if el.score > max_value:
            max_value = el.score
            max_index = i
    return max_index

class CustomMCTS():
    def __init__(self, tiles, board, strategy='max_depth'):

        self.initial_tiles, self.initial_board = tiles, board
        self.state = State(self.initial_board, self.initial_tiles)
        self.solution_checker = SolutionChecker(len(tiles), get_rows(board), get_cols(board))
        self.strategy=strategy
        self.n_tiles_placed = 0
        self.solution_tiles_order = []

        # for MCTS vis purposes
        self.colorful_states = True

    def predict(self, temp=1, N=3000):
        initial_state = self.state
        state = self.state
        available_tiles = state.tiles
        prev_state = state
        solution_found = False

        depth = 0
        self.val = 1
        while len(state.tiles):
            tile_placed = False
            states = []
            print(len(state.tiles))
            best_score = 0
            for i, tile in enumerate(state.tiles):
                success, new_board = SolutionChecker.get_next_turn(state, tile, self.val, destroy_state=False)
                self.n_tiles_placed += 1

                if success == ALL_TILES_USED:
                    print('solution found!')
                    solution_found = True
                    return initial_state, depth, solution_found
                if success == TILE_CANNOT_BE_PLACED:
                    # cannot place the tile.  this branch will not be considered
                    states.append(None)
                    continue
                else:
                    tile_placed = True
                    new_tiles = SolutionChecker.eliminate_pair_tiles(state.tiles, tile)
                    new_state = State(
                        board=new_board, tiles=new_tiles, parent=state)
                    state.children.append(new_state)
                    simulation_result, solution_tiles_order = self.perform_simulations(new_state, N=N)
                    if simulation_result == ALL_TILES_USED:
                        print('solution found in simulation!')
                        print(tile)
                        solution_found = True

                        # update scores of states found in simulation
                        new_state.score = len(state.tiles) / ORIENTATIONS - 1
                        _state = new_state.children[0]
                        while _state.children:
                            _state.score = (len(_state.tiles) / ORIENTATIONS) - 1
                            _state = _state.children[0]

                        if state.tile_placed:
                            self.solution_tiles_order.extend([state.tile_placed] + [tile] + solution_tiles_order)
                        else:
                            self.solution_tiles_order.extend([tile] + solution_tiles_order)
                        return initial_state, depth, solution_found
                    new_state.score = simulation_result
                    if new_state.score > best_score:
                        best_tile = tile
                        best_score = new_state.score
                    new_state.tile_placed = tile
                    state.solution_tiles_order.append(tile)
                    states.append(new_state)
            if not tile_placed:
                # no tile was placed, it's a dead end; end game
                return initial_state, depth, solution_found

            # PERFORMANCE:
            # for visualization this can be changed
            # all tiles will be 1 inside the frame for performance reasons
            self.val += 1

            depth += 1
            best_action = get_max_index(states) 
            prev_state = state
            new_state = states[best_action]
            print(best_tile, prev_state.tile_placed)
            if prev_state.tile_placed:
                self.solution_tiles_order.append(prev_state.tile_placed)


            state = new_state

        print('Solution found!')
        solution_found = True
        return initial_state, depth, solution_found


    def perform_simulations(self, state, N=3000):
        '''
        Given a state perform N simulations.
        One simulation consists of either filling container or having no more tiles to place.

        Returns average depth
        '''
        depths = []
        for n in range(N):
            depth, simulation_root_state, solution_tiles_order = self.perform_simulation(state.copy())
            if depth == ALL_TILES_USED:
                state.children = [simulation_root_state] #.children
                return ALL_TILES_USED, solution_tiles_order
            else:
                depths.append(depth)
        if self.strategy=='max_depth':
            _max = np.max(np.array(depths))
        elif self.strategy == 'avg_depth':
            _max = np.average(np.array(depths))
        return _max, None


    def perform_simulation(self, state):
        '''
        Performs the simulation until legal moves are available.
        If simulation ends by finding a solution, a root state starting from this simulation is returned
        '''

        solution_tiles_order = []
        depth = 0
        simulation_root_state = state  # in case simulation ends in solution; these states are the solution
        if len(state.tiles) == 0:
            print('perform_simulation called with empty tiles')
            return ALL_TILES_USED, simulation_root_state, solution_tiles_order
        val = self.val
        while True:
            val += 1
            if len(state.tiles) == 0:
                print('solution found in simulation')
                return ALL_TILES_USED, simulation_root_state, solution_tiles_order
            valid_moves = SolutionChecker.get_valid_next_moves(state, state.tiles, val=val, colorful_states=self.colorful_states)
            if not valid_moves:
                return depth, simulation_root_state, solution_tiles_order

            next_random_tile_index = random.randint(0, len(valid_moves) -1)
            success, new_board = SolutionChecker.get_next_turn(
                state, valid_moves[next_random_tile_index], val, destroy_state=True, 
                colorful_states=self.colorful_states
            )
            self.n_tiles_placed += 1
            solution_tiles_order.append(valid_moves[next_random_tile_index])

            if success == ALL_TILES_USED:
                print('grid is full')
                # no LFB on grid; probably means grid is full
                solution_tiles_order.append(valid_moves[next_random_tile_index])
                return ALL_TILES_USED, simulation_root_state, solution_tiles_order
            elif success == NO_NEXT_POSITION_TILES_UNUSED:
                print('no next position with unused tiles')
                return depth, simulation_root_state, solution_tiles_order
            elif success == TILE_CANNOT_BE_PLACED:
                # cannot place the tile. return depth reached
                return depth, simulation_root_state, solution_tiles_order
            else:
                new_tiles = SolutionChecker.eliminate_pair_tiles(state.tiles, valid_moves[next_random_tile_index])
                new_state = State(board=new_board, tiles=new_tiles, parent=state)

                new_state.score = -1  #  because no choice is performed for sequent actions
                state.children.append(new_state)
                state = new_state
            depth += 1
        return depth, simulation_root_state, solution_tiles_order


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s, state = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        # IGOR changed this behaviour

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        _s, state = self.game.stringRepresentation(canonicalBoard)

        if _s not in self.Es:
            self.Es[_s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[_s]!=0:
            # terminal node
            return self.Es[_s]

        if _s not in self.Ps:
            # leaf node
            self.Ps[_s], v = self.nnet.predict(canonicalBoard[0])
            valids = self.game.getValidMoves(canonicalBoard, 1)
            if valids.all():
                print(self.Ps[_s])
            self.Ps[_s] = self.Ps[_s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[_s])
            if sum_Ps_s > 0:
                self.Ps[_s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[_s] = self.Ps[_s] + valids
                self.Ps[_s] /= np.sum(self.Ps[_s])

            self.Vs[_s] = valids
            self.Ns[_s] = 0
            return v

        valids = self.Vs[_s]

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (_s,a) in self.Qsa:
                    u = self.Qsa[(_s,a)] + self.args.cpuct*self.Ps[_s][a]*math.sqrt(self.Ns[_s])/(1+self.Nsa[(_s,a)])
                    #if not state.state[0].any():
                    #    print(self.Qsa[(_s, a)], u, 'first', cur_best, u > cur_best, self.Nsa[(_s, a)])
                else:
                    # i Added + 100 - optimistic initialization??
                    u = self.args.cpuct*self.Ps[_s][a]*math.sqrt(self.Ns[_s] + EPS) + 100     # Q = 0 ?
                    #if not state.state[0].any():
                    #    print(u, 'second')

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player, next_vis_state = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (_s,a) in self.Qsa:
            self.Qsa[(_s,a)] = (self.Nsa[(_s,a)]*self.Qsa[(_s,a)] + v)/(self.Nsa[(_s,a)]+1)
            self.Nsa[(_s,a)] += 1

        else:
            self.Qsa[(_s,a)] = v
            self.Nsa[(_s,a)] = 1

        self.Ns[_s] += 1
        return v
