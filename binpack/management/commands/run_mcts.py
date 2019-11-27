from django.core.management.base import BaseCommand

import json
from binpack.MCTS import CustomMCTS
from binpack.models import Result
from data_generator import DataGenerator
# from asciitree import LeftAligned
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

def run_mcts():
    w = 31
    h = 31
    n = 21
    #n = 16
    #w = 10
    #h = 10

    N_simulations = config.n_sim
    from_file = False

    dg = DataGenerator(w, h)
    tiles, board = dg.gen_tiles_and_board(n, w, h, order_tiles=True, from_file=from_file)
    print(f'Starting problem with width {w}, height {h} and {n} tiles')
    print(f'TILES: {tiles}')
    print(f'Performing: {N_simulations} simulations per possible tile-action')

    custom_mcts = CustomMCTS(tiles, board)

    ret = custom_mcts.predict(N=N_simulations)
    child = ret.children[0]

    tree, all_nodes = ret.render_children(only_ids=True)


    k_v = {key: all_nodes[key].to_json() for key in all_nodes.keys()}

    from_file_str = ''
    if from_file:
        from_file_str = 'from_file'

    output_filename_base = f'{n}_{w}_{h}_{from_file_str}_{N_simulations}'
    k_v_json = json.dumps(k_v, cls=NpEncoder)

    tree_json = json.dumps(ret.render_to_json(), cls=NpEncoder)
    problem_generator = 'guillotine' if from_file else 'florian'
    with open(os.path.join(RESULTS_DIR, output_filename_base) + '_tree.json', 'w') as f:
        f.write(tree_json)
    Result.objects.create(
        rows=h,
        cols=w,
        tiles=tiles,
        result_Tree = tree_json,
        problem_generator=problem_generator,
        n_simulations=N_simulations,
    )


    print(tree_json)

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
        parser.add_argument('--n_sim', type=int, default=10, help='number of simulations from each action')

    def handle(self, *args, **options):
        print(args)
        run_mcts()
