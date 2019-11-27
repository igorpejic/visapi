from django.core.management.base import BaseCommand

import json
from binpack.MCTS import CustomMCTS
from binpack.models import Result
from data_generator import DataGenerator
from asciitree import LeftAligned
from collections import OrderedDict as OD

import os
import argparse

import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

RESULTS_DIR = 'results/'
np.random.seed(123) # reproducibility

ORIENTATIONS = 2
def run_mcts(tiles, rows, cols, n_sim):
    w = cols
    h = rows
    n = tiles
    #n = 16
    #w = 10
    #h = 10

    N_simulations = n_sim
    from_file = False

    dg = DataGenerator(w, h)
    tiles, board = dg.gen_tiles_and_board(n, w, h, order_tiles=True, from_file=from_file)
    print(f'Starting problem with width {w}, height {h} and {n} tiles')
    print(f'TILES: {tiles}')
    print(f'Performing: {N_simulations} simulations per possible tile-action')

    custom_mcts = CustomMCTS(tiles, board)

    ret, depth, solution_found = custom_mcts.predict(N=N_simulations)

    score = len(tiles) / ORIENTATIONS - depth
    child = ret.children[0]

    tree, all_nodes = ret.render_children(only_ids=True)


    k_v = {key: all_nodes[key].to_json() for key in all_nodes.keys()}

    from_file_str = ''
    if from_file:
        from_file_str = 'from_file'

    output_filename_base = f'{n}_{w}_{h}_{from_file_str}_{N_simulations}'
    k_v_json = json.dumps(k_v, cls=NpEncoder)

    tree_json = json.dumps(ret.render_to_json(), cls=NpEncoder)
    problem_generator = 'florian' if from_file else 'guillotine'
    with open(os.path.join(RESULTS_DIR, output_filename_base) + '_tree.json', 'w') as f:
        f.write(tree_json)
    Result.objects.create(
        rows=h,
        cols=w,
        tiles=tiles[:int(len(tiles)/2)],
        result_tree=tree_json,
        problem_generator=problem_generator,
        n_simulations=N_simulations,
        solution_found=solution_found,
        score=score
    )
    tree, all_nodes = ret.render_children(only_ids=False)
    tr = LeftAligned()
    res = tr(tree)
    print(res)

    output_filename = output_filename_base + '.txt'
    output_filename = os.path.join(RESULTS_DIR, output_filename)
    with open(output_filename, 'w') as f:
        f.write(res)
    return ret


class Command(BaseCommand):
    help = "Run mcts"

    def add_arguments(self, parser):
        parser.add_argument('tiles', type=int, default=10, help='number of tiles')
        parser.add_argument('rows', type=int, default=10, help='number of rows')
        parser.add_argument('cols', type=int, default=10, help='number of cols')
        parser.add_argument('--n_sim', type=int, default=10, help='number of simulations from each action')

    def handle(self, *args, **options):
        n_sim = options['n_sim']
        run_mcts(options['tiles'], options['rows'], options['cols'], n_sim)
