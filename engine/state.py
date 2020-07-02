import numpy as np
import json
from collections import OrderedDict
ORIENTATIONS = 2

class UUID(object):
    def __init__(self):
        self.i = 0

    def uuid(self):
        self.i += 1
        return self.i

_uuid = UUID()

def render_to_dict(node, tree=None, all_nodes={}, only_ids=False):

    all_nodes[node.uuid_str()] = node
    if not node.children:
        return {}, all_nodes

    if only_ids:
        node_str = node.uuid_str()
    else:
        node_str = str(node)

    if tree is None:
        tree = OrderedDict([])

    if node_str in tree:
        tree = tree[node_str]
        # all_nodes[node.uuid_str()] = node
    else:
        tree[node_str] = OrderedDict([])
        tree = tree[node_str]
    for child in node.children:
        if only_ids:
            child_str = child.uuid_str()
        else:
            child_str = str(child)
        tree[child_str], all_nodes = render_to_dict(child, tree, all_nodes, only_ids=only_ids)

    return tree, all_nodes

def render_to_json(node, tree=None, i=0, all_nodes={}, only_ids=False):

    node_dict = node.to_json()
    if not node.children:
        return node_dict

    if tree is None:
        tree = {}

    i+=1

    new_tree = {**node_dict}

    for child in node.children:
        new_tree['children'].append(render_to_json(child, i=i))

    return new_tree

class State(object):

    def __init__(self, board, tiles, parent=None, solution_tiles_order=None):
        self.board = np.copy(board)
        self.tiles = tiles[:]
        self.parent = parent
        self.uuid = _uuid.uuid()
        self.children = []
        self.score = None
        self.tile_placed = None
        if parent and parent.solution_tiles_order:
            self.solution_tiles_order = parent.solution_tiles_order
        else:
            self.solution_tiles_order = []

    def uuid_str(self):
      return f'{self.uuid}'


    def copy(self):
        return State(self.board, self.tiles, parent=self.parent)

    def child_with_biggest_score(self):
        return sorted(self.children, key=sort_key, reverse=True)[0]

    def render_to_json(self):
        return render_to_json(self)

    def render_children(self, only_ids=False):
        ret, all_nodes = render_to_dict(self, only_ids=only_ids)
        if only_ids:
            self_str = self.uuid_str()
        else:
            self_str = str(self)
        all_nodes[self.uuid_str()] = self
        return {self_str: ret}, all_nodes 

    def to_json(self):
        score = self.score or 0
        board = list(self.board)
        tiles = self.tiles
        tile_placed = self.tile_placed
        board = self.board.tolist()
        ret = {
            'tiles': tiles,
            'board': board,
            'tile_placed': tile_placed,
            'score': score,
            'name': self.uuid_str(),
            'children': [],
        }
        return ret

    def centralize_node_with_children(self):

        '''
        positions node with children in the middle of the search tree
        '''
        _state = self
        while _state.children:
            index_with_children = None
            for i, child in enumerate(_state.children):
                if child.children:
                    index_with_children = i
            if index_with_children is None:
                break
            if len(_state.children) % 2 == 0:
                middle_index = len(_state.children) // 2 - 1
            else:
                middle_index = len(_state.children) // 2
            next_state = _state.children[index_with_children]
            # print('middle index', middle_index, index_with_children, 'children:', len(state.children))
            _state.children[middle_index], _state.children[index_with_children] = _state.children[index_with_children], _state.children[middle_index]
            _state = next_state


    def __str__(self):
        return self.__repr__()

    def __repr__(self, short=False):
        if short:
            return f'{self.tiles}'

        output_list = [list(x) for x in self.tiles]
        if len(output_list) > 6:
            output_list = str(output_list[:6]) +  '(...)'

        output_board = ''
        if len(self.tiles) <= 4:
            output_board = self.board
        return f'({self.uuid}) Remaining tiles: {len(self.tiles) / ORIENTATIONS}, Tile placed: {self.tile_placed}. Tiles: {output_list}. Sim. depth:({self.score}) {output_board}'
