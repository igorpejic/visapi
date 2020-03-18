from django.core.management.base import BaseCommand

import json
import random
import pickle
import csv
from binpack.models import Result
from asciitree import LeftAligned
from collections import OrderedDict as OD
from cProfile import Profile

import os
import argparse
import uuid

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

class Command(BaseCommand):
    help = "Dump results to pickle"

    def add_arguments(self, parser):
        parser.add_argument('--output', action='store_true', help='output filename')
        parser.add_argument('--input', action='store_true', help='input filename')

    def handle(self, *args, **options):
        if options['input']:
            with open('results_dump.csv', 'rb') as csv_file:
                qs = pickle.load(csv_file)
                for q in qs:
                    r = Result.objects.filter(**dict(q))
                    if not r:
                        print('creating')
                        Result.objects.create(**q)
        elif options['output']:
            with open('results_dump.csv', 'wb') as csv_file:
                qs = Result.objects.all()
                fieldnames = ['rows', 'cols', 'n_simulations', 'tiles', 'n_tiles', 'score', 'solution_found',
                              'problem_generator', 'strategy', 'n_tiles_placed', 'their_id', 'their_tiles_placed',
                              'solution_tiles_order',
                              'problem_id', 'improved_sel']
                pickle.dump(qs.values(*fieldnames), csv_file)
