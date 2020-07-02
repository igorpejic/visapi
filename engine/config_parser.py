import argparse


def add_argument_group(name, parser, arg_lists):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

def str2bool(v):
  return v.lower() in ('true', '1')

def get_config():
    parser = argparse.ArgumentParser(description='Configuration file')
    arg_lists = []
    # Data
    data_arg = add_argument_group('Data', parser, arg_lists)
    data_arg.add_argument('--batch_size', type=int, default=256, help='batch size')
    data_arg.add_argument('--n', type=int, default=20, help='number of bins')
    data_arg.add_argument('--w', type=int, default=50, help='width of bin to fit in')
    data_arg.add_argument('--h', type=int, default=50, help='width of bin to fit in')
    data_arg.add_argument('--dimension', type=int, default=2, help='city dimension')
    data_arg.add_argument('--count_non_placed_tiles', action='store_true', help='Use number of non-placed tiles as reward')
    data_arg.add_argument('--combinatorial_reward', action='store_true', help='Use combinatorial reward where placing all tiles gives loss 0 otherwise 1.')

    # Network
    net_arg = add_argument_group('Network', parser, arg_lists)
    net_arg.add_argument('--input_embed', type=int, default=128, help='actor critic input embedding')
    net_arg.add_argument('--num_neurons', type=int, default=512, help='encoder inner layer neurons')
    net_arg.add_argument('--num_stacks', type=int, default=3, help='encoder num stacks')
    net_arg.add_argument('--num_heads', type=int, default=16, help='encoder num heads')
    net_arg.add_argument('--query_dim', type=int, default=360, help='decoder query space dimension')
    net_arg.add_argument('--num_units', type=int, default=256, help='decoder and critic attention product space')
    net_arg.add_argument('--num_neurons_critic', type=int, default=256, help='critic n-1 layer')

    # Train / test parameters
    train_arg = add_argument_group('Training', parser, arg_lists)
    train_arg.add_argument('--nb_steps', type=int, default=20000, help='nb steps')
    train_arg.add_argument('--init_B', type=float, default=7., help='critic init baseline')
    train_arg.add_argument('--lr_start', type=float, default=0.001, help='actor learning rate')
    train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='lr1 decay step')
    train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='lr1 decay rate')
    train_arg.add_argument('--temperature', type=float, default=1.0, help='pointer initial temperature')
    train_arg.add_argument('--freeze_first_batch', action='store_true', help='freeze the first random batch and reuse it')
    train_arg.add_argument('--C', type=float, default=10.0, help='pointer tanh clipping')
    train_arg.add_argument('--is_training', type=str2bool, default=True, help='switch to inference mode when model is trained') 

    config, unparsed = parser.parse_known_args()

    dir_ = str(config.dimension)+'D_'+'BBP'+str(config.n) +'_b'+str(config.batch_size)+'_e'+str(config.input_embed)+'_n'+str(config.num_neurons)+'_s'+str(config.num_stacks)+'_h'+str(config.num_heads)+ '_q'+str(config.query_dim) +'_u'+str(config.num_units)+'_c'+str(config.num_neurons_critic)+ '_lr'+str(config.lr_start)+'_d'+str(config.lr_decay_step)+'_'+str(config.lr_decay_rate)+ '_T'+str(config.temperature)+ '_steps'+str(config.nb_steps)+'_i'+str(config.init_B) 

    return config, unparsed, dir_

