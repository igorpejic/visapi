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
def run_mcts(options):
    cols = options['cols']
    rows = options['rows']
    n = options['n_tiles']
    from_file = options['from_file']
    n_sim = options['n_sim']
    strategy = 'avg_depth' if options['avg_depth'] else 'max_depth'

    dg = DataGenerator(cols, rows)
    instances = []
    if from_file:
        instances_from_file = dg.read_instances()
        for cols_rows, v in instances_from_file[n].items():
            for instance_from_file in v:
                instance = dg.transform_instance_visual_to_tiles_and_board(
                    cols_rows[1], cols_rows[0],
                    instance_from_file, order_tiles=True)
            instances.append(instance)
    else:
        n_problems_to_solve = 1
        for i in range(n_problems_to_solve):
            instances.append(dg.gen_tiles_and_board(
            n, cols, rows, order_tiles=True, from_file=from_file))


    for i, instance in enumerate(instances):
        # if i == 100:
        #     break
        print(instance)
        tiles, board = instance
        run_one_simulation(
            tiles, board, board.shape[1], board.shape[0], n_sim, from_file,
        strategy=strategy)


def run_one_simulation(tiles, board, cols, rows, n_sim, from_file, strategy='max_depth'):

    n = len(tiles) / ORIENTATIONS
    N_simulations = n_sim
    problem_generator = 'florian' if from_file else 'guillotine'
    print(f'Starting problem with rows {rows}, cols {cols} and {len(tiles) / 2} tiles')
    print(f'TILES: {tiles}')
    print(f'Performing: {N_simulations} simulations per possible tile-action')

    results = Result.objects.filter(
        rows=rows, cols=cols,
        tiles=tiles[:int(len(tiles)/ORIENTATIONS)],
        problem_generator=problem_generator,
        strategy=strategy
    )
    if results and from_file:
        print(f'Result already exists. Skipping. (results)')
        return

    custom_mcts = CustomMCTS(tiles, board, strategy=strategy)

    ret, depth, solution_found = custom_mcts.predict(N=N_simulations)

    score = len(tiles) / ORIENTATIONS - depth
    child = ret.children[0]

    tree, all_nodes = ret.render_children(only_ids=True)


    k_v = {key: all_nodes[key].to_json() for key in all_nodes.keys()}

    from_file_str = ''
    if from_file:
        from_file_str = 'from_file'

    output_filename_base = f'{n}_{cols}_{rows}_{from_file_str}_{N_simulations}'
    k_v_json = json.dumps(k_v, cls=NpEncoder)

    tree_json = json.dumps(ret.render_to_json(), cls=NpEncoder)
    with open(os.path.join(RESULTS_DIR, output_filename_base) + '_tree.json', 'w') as f:
        f.write(tree_json)
    Result.objects.create(
        rows=rows,
        cols=cols,
        tiles=tiles[:int(len(tiles)/ORIENTATIONS)],
        result_tree=tree_json,
        problem_generator=problem_generator,
        n_simulations=N_simulations,
        n_tiles=int(len(tiles) / ORIENTATIONS),
        solution_found=solution_found,
        score=score,
        strategy=strategy,
        n_tiles_placed=custom_mcts.n_tiles_placed
    )
    tree, all_nodes = ret.render_children(only_ids=False)
    tr = LeftAligned()
    res = tr(tree)
    print(res)
    print(f'Tiles placed: {custom_mcts.n_tiles_placed}')

    output_filename = output_filename_base + '.txt'
    output_filename = os.path.join(RESULTS_DIR, output_filename)
    with open(output_filename, 'w') as f:
        f.write(res)
    return ret


class Command(BaseCommand):
    help = "Run mcts"

    def add_arguments(self, parser):
        parser.add_argument('n_tiles', type=int, default=10, help='number of tiles')
        parser.add_argument('rows', type=int, default=10, help='number of rows')
        parser.add_argument('cols', type=int, default=10, help='number of cols')
        parser.add_argument('--n_sim', type=int, default=10, help='number of simulations from each action')
        parser.add_argument('--from_file', action='store_true', help='use instances from file')
        parser.add_argument('--avg_depth', action='store_true', help='avg_depth')

    def handle(self, *args, **options):
        run_mcts(options)
