from Coach import Coach
# from othello.pytorch.NNet import NNetWrapper as nn

# from othello.OthelloGame import OthelloGame as Game
# from othello.tensorflow.NNet import NNetWrapper as nn
# from binpack.tensorflow.NNet import NNetWrapper as nn
# from binpack.keras.ScalarKerasBinpackNNet import ScalarKerasBinpackNNet as nn
from binpack.keras.NNet import NNetWrapper as nn
from binpack.BinPackGame import BinPackGame as Game
from utils import *

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'numIters': 8,
    'numEps': 3,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 2,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 40,

})

if __name__=="__main__":
    N_TILES = 15 
    HEIGHT = 20
    WIDTH = 20
    g = Game(HEIGHT, WIDTH, N_TILES)

    nnet = nn(g, predict_v=True)

    if args.load_model and False:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
